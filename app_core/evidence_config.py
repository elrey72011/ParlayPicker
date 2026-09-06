"""Safe, actionable credential validation; never include credential values."""
import json
import os


class EvidenceConfigurationError(ValueError):
    """Only developer-authored, credential-free messages may use this exception."""


class EvidenceStorageError(ValueError):
    """Developer-authored integrity/operation errors containing no credentials."""


def service_account_info():
    raw = os.environ.get("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "")
    if not raw.strip():
        raise EvidenceConfigurationError(
            "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT is missing or empty. Add it as a root-level Streamlit secret.")
    try:
        info = json.loads(raw.lstrip("\ufeff"))
    except json.JSONDecodeError as exc:
        raise EvidenceConfigurationError(
            f"Service-account secret is not valid JSON (line {exc.lineno}, column {exc.colno}). "
            "In Streamlit Secrets, wrap the complete downloaded JSON in triple SINGLE quotes ('''), "
            "not triple double quotes. Preserve the JSON's backslash-n escapes in private_key. "
            "Do not paste a file path or only the private key.") from None
    required = ("client_email", "private_key", "token_uri")
    if not isinstance(info, dict) or info.get("type") != "service_account" or any(
            not isinstance(info.get(key), str) or not info[key].strip() for key in required):
        raise EvidenceConfigurationError(
            "The secret must contain the complete Google service-account JSON object, including "
            "type, client_email, private_key and token_uri. An OAuth client JSON file is not supported.")
    return info


def safe_error(exc, action):
    if isinstance(exc, (EvidenceConfigurationError, EvidenceStorageError)):
        return str(exc)
    return f"{action} failed ({type(exc).__name__}); check storage access/configuration."
