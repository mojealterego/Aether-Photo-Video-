# Aether security and privacy model

- Camera originals, faces, and biometric embeddings remain on the phone by default.
- Upload of a photo or face reference must require explicit consent, a configured trusted worker, encrypted transport and an option to delete the remote copy.
- Never expose ComfyUI HTTP/WebSocket endpoints directly to the public Internet without authentication/TLS or a private tunnel/VPN.
- Do not download or execute checkpoint files on behalf of an untrusted image or third-party job manifest.
- Use fixed work directories, path traversal validation, max tile sizes, and per-tile SHA-256 checks on inputs/outputs.
- Retain original image and mask independently from any AI-generated result; mark synthesized content as hypothetical, never forensically recovered.
- Full-face occlusion and purposeful anonymization cannot be reversed into ground-truth identity by inference.
- Report vulnerabilities privately to the repository maintainer; avoid including personal images or model credentials in issues.

Security design is a planned contract until independently tested.
