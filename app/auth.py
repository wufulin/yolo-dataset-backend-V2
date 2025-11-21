"""Authentication module for the YOLO dataset manager."""
import logging
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

security: HTTPBasic = HTTPBasic()


def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate user using HTTP Basic auth."""
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "admin")

    if not (correct_username and correct_password):
        logger.error(f"Failed authentication attempt for username: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username
