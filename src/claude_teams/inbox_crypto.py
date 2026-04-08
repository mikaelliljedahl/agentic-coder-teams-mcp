"""Optional inbox encryption helpers."""

import base64
import json
import os
from typing import cast

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


_MASTER_KEY_ENV = "CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY"
_HKDF_SALT = b"claude-teams-inbox-v1"


def encryption_enabled() -> bool:
    """Return whether inbox encryption is enabled.

    Returns:
        bool: True when the encryption master key env var is set.

    """
    return bool(os.environ.get(_MASTER_KEY_ENV))


def _derive_fernet(team_name: str) -> Fernet:
    master_key = os.environ.get(_MASTER_KEY_ENV)
    if not master_key:
        raise RuntimeError(
            f"Inbox encryption key is required for this inbox, but {_MASTER_KEY_ENV} is not set."
        )

    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_HKDF_SALT,
        info=team_name.encode(),
    ).derive(master_key.encode())
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_entry(team_name: str, payload: dict[str, object]) -> dict[str, object]:
    """Encrypt an inbox entry when encryption is enabled.

    Args:
        team_name (str): Team name used for key derivation.
        payload (dict[str, object]): Plaintext inbox payload.

    Returns:
        dict[str, object]: Encrypted envelope or the original payload.

    """
    if not encryption_enabled():
        return payload

    token = _derive_fernet(team_name).encrypt(
        json.dumps(payload, separators=(",", ":")).encode()
    )
    return {
        "enc": {
            "v": 1,
            "alg": "fernet",
            "token": token.decode(),
        }
    }


def decrypt_entry(team_name: str, payload: dict[str, object]) -> dict[str, object]:
    """Decrypt an inbox entry when it contains encrypted content.

    Args:
        team_name (str): Team name used for key derivation.
        payload (dict[str, object]): Stored inbox payload.

    Returns:
        dict[str, object]: Plaintext inbox payload.

    """
    encrypted = payload.get("enc")
    if not isinstance(encrypted, dict):
        return payload
    encrypted_dict = cast(dict[str, object], encrypted)

    token = encrypted_dict.get("token")
    if not isinstance(token, str):
        raise RuntimeError("Malformed encrypted inbox entry: missing ciphertext token.")

    try:
        plaintext = _derive_fernet(team_name).decrypt(token.encode())
    except InvalidToken as exc:
        raise RuntimeError(
            "Unable to decrypt inbox entry with the configured encryption key."
        ) from exc

    data = json.loads(plaintext)
    if not isinstance(data, dict):
        raise RuntimeError("Decrypted inbox entry must be a JSON object.")
    return data
