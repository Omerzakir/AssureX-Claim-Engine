"""Firebase ID-token authorization. Client-provided roles/UIDs are never trusted."""
from __future__ import annotations
import os
from functools import lru_cache
from fastapi import Depends, HTTPException, Request

@lru_cache(maxsize=1)
def firebase_clients():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, auth
        if not firebase_admin._apps:
            credential_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if credential_path:
                firebase_admin.initialize_app(credentials.Certificate(credential_path))
            else:
                firebase_admin.initialize_app()  # Application Default Credentials
        return auth, firestore.client()
    except Exception as exc:
        raise HTTPException(503, detail='Server authentication is not configured.') from exc

def identity(request: Request):
    header = request.headers.get('authorization', '')
    if not header.startswith('Bearer ') or not header[7:].strip():
        raise HTTPException(401, detail='Authentication required.')
    auth, db = firebase_clients()
    try:
        token = auth.verify_id_token(header[7:].strip(), check_revoked=True)
        uid = token['uid']
        snapshot = db.collection('users').document(uid).get()
        if not snapshot.exists:
            raise HTTPException(403, detail='Account profile unavailable.')
        profile = snapshot.to_dict() or {}
        if profile.get('isActive') is False:
            raise HTTPException(403, detail='Account disabled.')
        role = str(profile.get('role', 'user')).lower()
        if role not in ('user', 'reviewer', 'employee', 'admin'):
            raise HTTPException(403, detail='Invalid account role.')
        return {'uid': uid, 'role': role, 'email': token.get('email'), 'db': db}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(401, detail='Invalid or expired authentication.') from exc

def require_role(*roles):
    def dependency(user=Depends(identity)):
        if user['role'] not in roles:
            raise HTTPException(403, detail='Insufficient permissions.')
        return user
    return dependency
