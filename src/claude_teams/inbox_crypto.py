"""Optional inbox encryption helpers."""

import base64
import json
import os
from typing import cast

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from claude_teams.errors import (
    DecryptedInboxNotObjectError,
    InboxDecryptError,
    InboxEncryptionKeyMissingError,
    InboxMasterKeyTooShortError,
    MalformedEncryptedInboxEntryError,
)

_MASTER_KEY_ENV = "CLAUDE_TEAMS_ENCRYPTION_MASTER_KEY"
_HKDF_SALT = b"claude-teams-inbox-v1"
_MIN_MASTER_KEY_LEN = 32


def encryption_enabled() -> bool:
    """Return whether inbox encryption is enabled.

    Returns:
        bool: True when the encryption master key env var is set.

    """
    return bool(os.environ.get(_MASTER_KEY_ENV))


def _derive_fernet(team_name: str) -> Fernet:
    master_key = os.environ.get(_MASTER_KEY_ENV)
    if not master_key:
        raise InboxEncryptionKeyMissingError(_MASTER_KEY_ENV)
    if len(master_key) < _MIN_MASTER_KEY_LEN:
        raise InboxMasterKeyTooShortError(_MASTER_KEY_ENV, _MIN_MASTER_KEY_LEN)

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
        raise MalformedEncryptedInboxEntryError()

    try:
        plaintext = _derive_fernet(team_name).decrypt(token.encode())
    except InvalidToken as exc:
        raise InboxDecryptError() from exc

    data = json.loads(plaintext)
    if not isinstance(data, dict):
        raise DecryptedInboxNotObjectError()
    return data
