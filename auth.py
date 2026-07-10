import os
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-key-in-env-file")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 din tak token valid rahega

# ---------------------------------------------------------
# PASSWORD HASHING
# ---------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Plain password ko hash (encrypt) karna, taake Redis mein safe rahe"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Login ke waqt: user ne jo password diya, wo hash se match karta hai ya nahi"""
    return pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------
# JWT TOKEN CREATE / VERIFY
# ---------------------------------------------------------
def create_access_token(data: dict) -> str:
    """Login successful hone par ek naya JWT token banana"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Token ko verify karna aur uske andar ka data (user email) nikalna"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None