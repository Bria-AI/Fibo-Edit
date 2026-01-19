import os


def validate_vertex_env():
    """
    Validates that the required Vertex AI environment variables are set and valid.
    Raises EnvironmentError, ValueError, or FileNotFoundError if validation fails.
    """
    required_vars = [
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS"
    ]
    
    # Check for existence
    missing = [v for v in required_vars if v not in os.environ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    # Check for empty values
    empty = [v for v in required_vars if not os.environ.get(v, "").strip()]
    if empty:
        raise ValueError(
            f"The following environment variables are defined but empty: {', '.join(empty)}"
        )