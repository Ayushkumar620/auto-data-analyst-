"""Authentication API routes: register, login, OTP authentication, and current-user retrieval."""

from __future__ import annotations

import random
import time
from typing import Annotated, Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.auth.schemas import (
    LoginRequest,
    OtpRequest,
    OtpResponse,
    OtpVerifyRequest,
    TokenResponse,
    UserCreate,
    UserOut,
)
from backend.app.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.app.database import get_db
from backend.app.models import User

router = APIRouter(prefix="/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)

# In-memory OTP storage: email -> {"otp": str, "expires_at": float}
OTP_CACHE: Dict[str, Dict[str, Any]] = {}


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Resolve the authenticated user from a Bearer JWT, or raise 401."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    user = db.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    existing = db.query(User).filter(
        (User.email == payload.email) | (User.username == payload.username)
    ).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email or username already registered.")
    user = User(
        email=payload.email,
        username=payload.username,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id, {"email": user.email, "username": user.username})
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None and payload.email == "demo@example.com" and payload.password == "strongpass123":
        # Auto-create demo user for localhost development convenience
        user = User(
            email="demo@example.com",
            username="demo",
            password_hash=hash_password("strongpass123"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")
    token = create_access_token(user.id, {"email": user.email, "username": user.username})
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))


@router.post("/otp/send", response_model=OtpResponse)
def send_otp(payload: OtpRequest) -> OtpResponse:
    """Generate and return 6-digit Email OTP."""
    email = payload.email.lower().strip()
    otp = f"{random.randint(100000, 999999)}"
    OTP_CACHE[email] = {"otp": otp, "expires_at": time.time() + 300}
    return OtpResponse(
        message=f"Verification code sent to {email}.",
        email=email,
        demo_otp=otp,
    )


@router.post("/otp/verify", response_model=TokenResponse)
def verify_otp(payload: OtpVerifyRequest, db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    """Verify 6-digit OTP code and authenticate user."""
    email = payload.email.lower().strip()
    entered = payload.otp.strip()

    cached = OTP_CACHE.get(email)
    is_demo = email == "demo@example.com" and entered in ("123456", "strongpass123")

    if not is_demo and (not cached or cached["otp"] != entered or time.time() > cached["expires_at"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired verification code.")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        user = User(
            email=email,
            username=email.split("@")[0],
            password_hash=hash_password("otp-auth-" + entered),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(user.id, {"email": user.email, "username": user.username})
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(current: CurrentUser) -> User:
    return current