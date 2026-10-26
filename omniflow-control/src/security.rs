#![forbid(unsafe_code)]
// Native OmniFlow security primitives. These are not a complete identity
// provider or durable KMS/HSM deployment (SYS-AETHERIS-OMNIFLOW-016).
use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use chrono::{DateTime, SecondsFormat, Utc};
use parking_lot::RwLock;
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use thiserror::Error;
use url::Url;
use uuid::Uuid;
use zeroize::Zeroizing;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SecurityError {
    #[error("model download URL denied")]
    UrlDenied,
    #[error("credential not found or not authorized")]
    CredentialDenied,
    #[error("envelope encryption failure")]
    EncryptionFailed,
}

/// Caller must additionally pin and validate all resolved DNS IPs and
/// redirects at connection time. Static URL checks alone cannot stop DNS rebinding.
pub struct ModelDownloadGuard { allowlisted_hosts: HashSet<String> }
impl ModelDownloadGuard {
    pub fn new(hosts: impl IntoIterator<Item=String>) -> Self {
        Self { allowlisted_hosts: hosts.into_iter().map(|host|host.to_lowercase()).collect() }
    }
    pub fn validate(&self, value:&str) -> Result<Url,SecurityError> {
        let url=Url::parse(value).map_err(|_| SecurityError::UrlDenied)?;
        if url.scheme()!="https" || url.fragment().is_some()
            || !url.username().is_empty() || url.password().is_some()
            || url.port_or_known_default()!=Some(443) {
            return Err(SecurityError::UrlDenied)
        }
        let host=url.host_str().ok_or(SecurityError::UrlDenied)?.to_lowercase();
        if host.parse::<IpAddr>().is_ok() || host=="localhost"
            || host.ends_with(".localhost")
            || !self.allowlisted_hosts.contains(&host) {
            return Err(SecurityError::UrlDenied)
        }
        Ok(url)
    }
}

struct SecretEntry {
    tenant_id:String,
    provider:String,
    encrypted_dek:Vec<u8>,
    dek_nonce:[u8;12],
    encrypted_value:Vec<u8>,
    value_nonce:[u8;12],
    revoked:bool,
}

/// Local in-memory DEK/KEK vault. Production requires persistent encrypted
/// records and master KEK in external KMS/HSM, never process configuration.
pub struct SecretBrokerVault {
    master_kek:Zeroizing<[u8;32]>,
    entries:RwLock<HashMap<String,SecretEntry>>,
}
impl SecretBrokerVault {
    pub fn new(master_kek:[u8;32]) -> Self {
        Self {master_kek:Zeroizing::new(master_kek),entries:RwLock::new(HashMap::new())}
    }

    pub fn enroll(&self,tenant_id:&str,provider:&str,
                  credential:&str) -> Result<String,SecurityError> {
        if tenant_id.trim().is_empty() || provider.trim().is_empty()
            || credential.is_empty() {return Err(SecurityError::CredentialDenied)}
        let mut dek=Zeroizing::new([0u8;32]);
        let mut dek_nonce=[0u8;12];
        let mut value_nonce=[0u8;12];
        OsRng.fill_bytes(&mut dek[..]);
        OsRng.fill_bytes(&mut dek_nonce);
        OsRng.fill_bytes(&mut value_nonce);

        let cipher=Aes256Gcm::new_from_slice(&dek[..])
            .map_err(|_| SecurityError::EncryptionFailed)?;
        let encrypted_value=cipher.encrypt(Nonce::from_slice(&value_nonce),credential.as_bytes())
            .map_err(|_| SecurityError::EncryptionFailed)?;
        let wrap=Aes256Gcm::new_from_slice(&self.master_kek[..])
            .map_err(|_| SecurityError::EncryptionFailed)?;
        let encrypted_dek=wrap.encrypt(Nonce::from_slice(&dek_nonce),&dek[..])
            .map_err(|_| SecurityError::EncryptionFailed)?;

        let handle=format!("cred_hnd_{}",Uuid::new_v4().simple());
        let entry=SecretEntry {tenant_id:tenant_id.into(),provider:provider.into(),
            encrypted_dek,dek_nonce,encrypted_value,value_nonce,revoked:false};
        self.entries.write().insert(handle.clone(),entry);
        Ok(handle)
    }

    pub fn resolve(&self,handle:&str,tenant_id:&str,provider:&str)
                   -> Result<Zeroizing<String>,SecurityError> {
        let entries=self.entries.read();
        let entry=entries.get(handle).ok_or(SecurityError::CredentialDenied)?;
        if entry.revoked || entry.tenant_id!=tenant_id || entry.provider!=provider {
            return Err(SecurityError::CredentialDenied)
        }
        let wrap=Aes256Gcm::new_from_slice(&self.master_kek[..])
            .map_err(|_| SecurityError::EncryptionFailed)?;
        let dek=Zeroizing::new(wrap.decrypt(
            Nonce::from_slice(&entry.dek_nonce),entry.encrypted_dek.as_ref())
            .map_err(|_| SecurityError::CredentialDenied)?);
        let cipher=Aes256Gcm::new_from_slice(&dek)
            .map_err(|_| SecurityError::EncryptionFailed)?;
        let plaintext=Zeroizing::new(cipher.decrypt(
            Nonce::from_slice(&entry.value_nonce),entry.encrypted_value.as_ref())
            .map_err(|_| SecurityError::CredentialDenied)?);
        let value=String::from_utf8(plaintext.to_vec())
            .map_err(|_| SecurityError::CredentialDenied)?;
        Ok(Zeroizing::new(value))
    }

    pub fn revoke(&self,handle:&str,tenant_id:&str)->Result<(),SecurityError> {
        let mut entries=self.entries.write();
        let entry=entries.get_mut(handle).ok_or(SecurityError::CredentialDenied)?;
        if entry.tenant_id!=tenant_id {return Err(SecurityError::CredentialDenied)}
        entry.revoked=true;
        Ok(())
    }
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct AuditEvent {
    pub timestamp:DateTime<Utc>,
    pub actor_id:String,
    pub tenant_pseudonym:String,
    pub action:String,
    pub resource_digest:String,
    pub decision:String,
    pub trace_id:String,
    pub previous_event_hash:String,
    pub event_hash:String,
}
fn hash_fields(values:&[&str])->String {
    let mut hasher=Sha256::new();
    hasher.update(b"AETHERIS_AUDIT_V1:");
    for value in values {
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value.as_bytes());
    }
    format!("sha256:{}",hex::encode(hasher.finalize()))
}
impl AuditEvent {
    fn calculate_hash(&self)->String {
        let time=self.timestamp.to_rfc3339_opts(SecondsFormat::Nanos,true);
        hash_fields(&[&time,&self.actor_id,&self.tenant_pseudonym,&self.action,
            &self.resource_digest,&self.decision,&self.trace_id,&self.previous_event_hash])
    }
}
#[derive(Default)]
pub struct TamperEvidentAuditLog { chain:RwLock<Vec<AuditEvent>> }
impl TamperEvidentAuditLog {
    pub fn new()->Self {Self::default()}
    pub fn append(&self,actor_id:&str,tenant_id:&str,action:&str,
                  resource_digest:&str,decision:&str,trace_id:&str)->AuditEvent {
        let mut chain=self.chain.write();
        let previous_event_hash=chain.last().map_or_else(
            ||format!("sha256:{}", "0".repeat(64)),|last|last.event_hash.clone());
        let tenant_pseudonym=hash_fields(&["TENANT",tenant_id]);
        let mut event=AuditEvent {timestamp:Utc::now(),actor_id:actor_id.into(),
            tenant_pseudonym,action:action.into(),resource_digest:resource_digest.into(),
            decision:decision.into(),trace_id:trace_id.into(),
            previous_event_hash,event_hash:String::new()};
        event.event_hash=event.calculate_hash();
        chain.push(event.clone());
        event
    }
    pub fn verify(&self)->bool {
        let chain=self.chain.read();
        let mut previous=format!("sha256:{}", "0".repeat(64));
        for event in chain.iter() {
            if event.previous_event_hash!=previous
                || event.event_hash!=event.calculate_hash() {return false}
            previous=event.event_hash.clone();
        }
        true
    }
    pub fn snapshot(&self)->Vec<AuditEvent>{self.chain.read().clone()}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disallow_hostile_model_download_url() {
        let guard=ModelDownloadGuard::new(["models.example.org".to_string()]);
        assert!(guard.validate("https://models.example.org/release/model.safetensors").is_ok());
        for bad in ["http://models.example.org/model","https://127.0.0.1/model",
            "https://localhost/model","https://attacker.example/model",
            "https://models.example.org:444/model",
            "https://admin:password@models.example.org/model",
            "https://models.example.org/model#fragment"] {
            assert_eq!(guard.validate(bad).unwrap_err(),SecurityError::UrlDenied);
        }
    }
    #[test]
    fn secret_handles_are_tenant_scoped_and_revocable() {
        let vault=SecretBrokerVault::new([23;32]);
        let handle=vault.enroll("tenant-a","provider","private-key").unwrap();
        assert!(handle.starts_with("cred_hnd_"));
        assert_eq!(vault.resolve(&handle,"tenant-a","provider").unwrap().as_str(),"private-key");
        assert_eq!(vault.resolve(&handle,"tenant-b","provider").unwrap_err(),SecurityError::CredentialDenied);
        assert_eq!(vault.resolve(&handle,"tenant-a","wrong").unwrap_err(),SecurityError::CredentialDenied);
        vault.revoke(&handle,"tenant-a").unwrap();
        assert_eq!(vault.resolve(&handle,"tenant-a","provider").unwrap_err(),SecurityError::CredentialDenied);
    }
    #[test]
    fn audit_verifies_event_contents_and_chain() {
        let audit=TamperEvidentAuditLog::new();
        let a=audit.append("actor","tenant","submit","sha256:resource","ALLOW","trace1");
        let b=audit.append("actor","tenant","verify","sha256:result","PASS","trace2");
        assert_eq!(b.previous_event_hash,a.event_hash);
        assert!(audit.verify());
        assert_eq!(audit.snapshot().len(),2);
    }
}
