//! GLB container: parse into chunks, write back byte for byte, read accessors.
use serde_json::Value;

pub const CHUNK_JSON: u32 = 0x4E4F_534A;
pub const CHUNK_BIN: u32 = 0x004E_4942;

pub struct Chunk {
    pub kind: u32,
    pub data: Vec<u8>,
}

pub struct Glb {
    pub version: u32,
    pub chunks: Vec<Chunk>,
    pub json: Value,
}

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// One accessor read into f64 values, `ncomp` per element.
pub struct Accessor {
    pub count: usize,
    pub ncomp: usize,
    pub data: Vec<f64>,
}

impl Accessor {
    pub fn get(&self, i: usize, c: usize) -> f64 {
        self.data[i * self.ncomp + c]
    }
}

impl Glb {
    pub fn parse(bytes: &[u8]) -> Result<Glb, String> {
        if bytes.len() < 12 || &bytes[0..4] != b"glTF" {
            return Err("not a GLB file (missing glTF magic)".into());
        }
        let version = u32le(bytes, 4);
        let total = u32le(bytes, 8) as usize;
        if total != bytes.len() {
            return Err(format!("header says {total} bytes, file has {}", bytes.len()));
        }
        let mut chunks = Vec::new();
        let mut off = 12;
        while off < total {
            if off + 8 > total {
                return Err(format!("truncated chunk header at byte {off}"));
            }
            let len = u32le(bytes, off) as usize;
            let kind = u32le(bytes, off + 4);
            let end = off + 8 + len;
            if end > total {
                return Err(format!("chunk at byte {off} runs past the end"));
            }
            chunks.push(Chunk { kind, data: bytes[off + 8..end].to_vec() });
            off = end;
        }
        let jc = chunks.first().filter(|c| c.kind == CHUNK_JSON).ok_or("first chunk is not JSON")?;
        let mut end = jc.data.len();
        while end > 0 && (jc.data[end - 1] == b' ' || jc.data[end - 1] == 0) {
            end -= 1;
        }
        let json: Value = serde_json::from_slice(&jc.data[..end]).map_err(|e| format!("JSON chunk: {e}"))?;
        Ok(Glb { version, chunks, json })
    }

    /// Writes the container from its parsed chunks. With no edits this equals the input file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let total: usize = 12 + self.chunks.iter().map(|c| 8 + c.data.len()).sum::<usize>();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        for c in &self.chunks {
            out.extend_from_slice(&(c.data.len() as u32).to_le_bytes());
            out.extend_from_slice(&c.kind.to_le_bytes());
            out.extend_from_slice(&c.data);
        }
        out
    }

    pub fn bin(&self) -> &[u8] {
        self.chunks.iter().find(|c| c.kind == CHUNK_BIN).map(|c| c.data.as_slice()).unwrap_or(&[])
    }

    pub fn list(&self, key: &str) -> &[Value] {
        self.json.get(key).and_then(Value::as_array).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Bytes of a buffer view (buffer 0 = the BIN chunk).
    pub fn view_bytes(&self, view: usize) -> Result<&[u8], String> {
        let v = self.list("bufferViews").get(view).ok_or(format!("bufferView {view} missing"))?;
        if v.get("buffer").and_then(Value::as_u64).unwrap_or(0) != 0 {
            return Err("only buffer 0 (the BIN chunk) is supported".into());
        }
        let off = v.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let len = v.get("byteLength").and_then(Value::as_u64).ok_or("bufferView without byteLength")? as usize;
        self.bin().get(off..off + len).ok_or(format!("bufferView {view} outside the BIN chunk"))
    }

    pub fn accessor(&self, i: usize) -> Result<Accessor, String> {
        let a = self.list("accessors").get(i).ok_or(format!("accessor {i} missing"))?;
        if a.get("sparse").is_some() {
            return Err(format!("accessor {i} is sparse (not supported)"));
        }
        let count = a.get("count").and_then(Value::as_u64).ok_or("accessor without count")? as usize;
        let ncomp = match a.get("type").and_then(Value::as_str).unwrap_or("") {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            t => return Err(format!("accessor {i}: type {t} not supported")),
        };
        let ct = a.get("componentType").and_then(Value::as_u64).unwrap_or(0);
        let size = match ct {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => return Err(format!("accessor {i}: componentType {ct}")),
        };
        let normalized = a.get("normalized").and_then(Value::as_bool).unwrap_or(false);
        let view = a.get("bufferView").and_then(Value::as_u64).ok_or("accessor without bufferView")? as usize;
        let vb = self.view_bytes(view)?;
        let stride = self.list("bufferViews")[view].get("byteStride").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(size * ncomp);
        let base = a.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let mut data = Vec::with_capacity(count * ncomp);
        for e in 0..count {
            for c in 0..ncomp {
                let o = base + e * stride + c * size;
                let b = vb.get(o..o + size).ok_or(format!("accessor {i} reads past its bufferView"))?;
                let v = match ct {
                    5126 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                    5125 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                    5123 => {
                        let x = u16::from_le_bytes([b[0], b[1]]) as f64;
                        if normalized { x / 65535.0 } else { x }
                    }
                    5122 => {
                        let x = i16::from_le_bytes([b[0], b[1]]) as f64;
                        if normalized { (x / 32767.0).max(-1.0) } else { x }
                    }
                    5121 => {
                        let x = b[0] as f64;
                        if normalized { x / 255.0 } else { x }
                    }
                    _ => {
                        let x = b[0] as i8 as f64;
                        if normalized { (x / 127.0).max(-1.0) } else { x }
                    }
                };
                data.push(v);
            }
        }
        Ok(Accessor { count, ncomp, data })
    }
}
