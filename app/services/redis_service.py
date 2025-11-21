"""Redis service providing client helpers and upload session management."""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import redis.asyncio as redis
from redis.asyncio.client import Redis
from redis.exceptions import ConnectionError, ResponseError, TimeoutError

from app.config import settings
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class RedisSessionManager:
    """Manager responsible for upload sessions stored in Redis."""

    def __init__(self):
        """Initialize Redis client references and configuration."""
        self.redis: Optional[Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Session configuration
        self.session_ttl = settings.redis_session_ttl
        self.lock_timeout = settings.redis_session_lock_timeout
        self.session_key_prefix = "upload:session:"
        self.chunk_key_prefix = "upload:chunk:"
        self.lock_key_prefix = "upload:lock:"
        self.lock_script_sha: Optional[str] = None

    async def connect(self) -> None:
        """Connect to Redis (single-node configuration)."""
        try:
            # Prefer redis_url when provided, otherwise fall back to host/port
            redis_url_value = getattr(settings, 'redis_url', None) or ""
            redis_url_value = redis_url_value.strip() if redis_url_value else ""
            
            # Use URL if defined and not the default localhost DSN;
            # even localhost should use URL if user explicitly configures it
            use_url = bool(redis_url_value and redis_url_value != "redis://localhost:6379/0")
            
            # Emit config details for troubleshooting
            logger.info(
                "Redis configuration: redis_url=%s, redis_host=%s, redis_port=%s, using_url=%s",
                redis_url_value[:80] if redis_url_value else 'not set (using defaults)',
                settings.redis_host,
                settings.redis_port,
                use_url,
            )
            
            if use_url:
                # Build client from URL (single-node Redis)
                # Mask password in logs
                url_for_log = redis_url_value.split('@')[-1] if '@' in redis_url_value else redis_url_value
                logger.info("Connecting to Redis via URL: %s", url_for_log)
                self.redis = redis.from_url(
                    redis_url_value,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=settings.redis_max_connections,
                    socket_timeout=settings.redis_connection_pool_timeout,
                    socket_connect_timeout=settings.redis_connection_pool_timeout,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
            else:
                # Host/port connection (single-node)
                logger.info("Connecting to Redis via host/port %s:%s", settings.redis_host, settings.redis_port)
                # Create connection pool
                self._connection_pool = redis.ConnectionPool(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    password=settings.redis_password,
                    max_connections=settings.redis_max_connections,
                    socket_timeout=settings.redis_connection_pool_timeout,
                    socket_connect_timeout=settings.redis_connection_pool_timeout,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

                # Create Redis client
                self.redis = redis.Redis(
                    connection_pool=self._connection_pool,
                    encoding="utf-8",
                    decode_responses=True,
                )

            # Verify connectivity
            await self.redis.ping()
            
            # Ensure instance is writable (single-node Redis should allow writes)
            test_key = f"__test_write_{int(time.time())}"
            try:
                await self.redis.set(test_key, "test", ex=1)  # expire after 1 second
                await self.redis.delete(test_key)
                logger.info("Redis connection established (single-node, writable)")
            except ResponseError as e:
                error_msg = str(e).lower()
                if "read only" in error_msg or "readonly" in error_msg:
                    error_msg_full = (
                        "Redis connection failed: connected to read-only replica and cannot write.\n"
                        "Single-node Redis should always be writable.\n"
                        "Please verify:\n"
                        f"  1. Configuration correctness: redis_url={redis_url_value if use_url else 'N/A'}, "
                        f"host={settings.redis_host}, port={settings.redis_port}\n"
                        "  2. Redis server health\n"
                        "  3. Whether you accidentally connected to a replica\n"
                        "  4. Whether redis.conf enables read-only mode (check 'replica-read-only'/'slave-read-only')\n"
                        "  5. For Redis clusters, ensure connections target the primary node"
                    )
                    logger.error(error_msg_full)
                    # Clean up resources
                    if self.redis:
                        await self.redis.close()
                    if self._connection_pool:
                        await self._connection_pool.disconnect()
                    raise ConnectionError(error_msg_full)
                # Other errors bubble up after logging
                logger.warning("Redis write test failed (connection established): %s", e)
                raise

            # Preload Lua scripts
            await self._load_lua_scripts()

            # Start background cleanup task
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            # Clean up resources
            if self.redis:
                try:
                    await self.redis.close()
                except:
                    pass
            if self._connection_pool:
                try:
                    await self._connection_pool.disconnect()
                except:
                    pass
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis and stop background tasks."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self.redis:
            await self.redis.close()

        if self._connection_pool:
            await self._connection_pool.disconnect()

        logger.info("Redis connection closed")

    async def _load_lua_scripts(self):
        """Load Lua scripts used for distributed locking."""
        try:
            # Distributed session lock script
            lock_script = """
            -- KEYS[1]: Lock key name
            -- ARGV[1]: Lock timeout (seconds) - must be an integer
            -- ARGV[2]: Request ID (unique client identifier)

            local timeout = tonumber(ARGV[1])
            if timeout == nil or timeout <= 0 then
                return redis.error_reply("Invalid timeout value")
            end
            
            if redis.call("SET", KEYS[1], ARGV[2], "EX", timeout, "NX") then
                return 1
            else
                return 0
            end
            """

            # Unlock script
            unlock_script = """
            -- KEYS[1]: Lock key name
            -- ARGV[1]: Request ID
            -- ARGV[2]: Session ID

            local lock_value = redis.call("GET", KEYS[1])
            if lock_value == ARGV[1] then
                redis.call("DEL", KEYS[1])
                -- Remove session state
                redis.call("DEL", "upload:session:" .. ARGV[2])
                return 1
            else
                return 0
            end
            """

            # Register scripts with Redis and store SHAs
            self.lock_script_sha = await self.redis.script_load(lock_script)
            self.unlock_script_sha = await self.redis.script_load(unlock_script)

            logger.info("Redis Lua scripts loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Lua scripts: {e}")
            raise

    def _get_session_key(self, upload_id: str) -> str:
        """Return Redis key for the upload session hash."""
        return f"{self.session_key_prefix}{upload_id}"

    def _get_chunk_key(self, upload_id: str, chunk_index: int) -> str:
        """Return Redis key for storing chunk metadata."""
        return f"{self.chunk_key_prefix}{upload_id}:{chunk_index}"

    def _get_lock_key(self, upload_id: str) -> str:
        """Return Redis key for the distributed lock."""
        return f"{self.lock_key_prefix}{upload_id}"

    async def create_session(
        self,
        upload_id: str,
        filename: str,
        total_size: int,
        total_chunks: int,
        chunk_size: int,
        temp_dir: str,
        temp_file: str,
        user_id: str = None
    ) -> bool:
        """Create a new upload session record in Redis."""
        try:
            session_data = {
                "upload_id": upload_id,
                "filename": filename,
                "total_size": total_size,
                "total_chunks": total_chunks,
                "chunk_size": chunk_size,
                "received_chunks": json.dumps([]),
                "temp_dir": temp_dir,
                "temp_file": temp_file,
                "user_id": user_id,
                "status": "uploading",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(seconds=self.session_ttl)).isoformat()
            }

            # Use pipeline for atomic multi-command execution
            pipe = self.redis.pipeline()
            pipe.hset(self._get_session_key(upload_id), mapping=session_data)
            pipe.expire(self._get_session_key(upload_id), self.session_ttl)
            pipe.zadd("upload:session:index", {upload_id: time.time()})
            pipe.expire("upload:session:index", self.session_ttl)

            await pipe.execute()

            logger.info(f"Upload session created: {upload_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create session {upload_id}: {e}")
            return False

    async def get_session(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Fetch upload session metadata."""
        try:
            session_data_raw = await self.redis.hgetall(self._get_session_key(upload_id))

            if not session_data_raw:
                return None

            # Convert bytes keys/values to str (hgetall may still return bytes)
            session_data = {}
            for key, value in session_data_raw.items():
                # Decode key if needed
                if isinstance(key, bytes):
                    key_str = key.decode('utf-8')
                else:
                    key_str = str(key)
                
                # Decode value if needed
                if isinstance(value, bytes):
                    value_str = value.decode('utf-8')
                else:
                    value_str = str(value) if value is not None else None
                
                session_data[key_str] = value_str

            # Cast numeric fields to integers
            numeric_fields = ["total_size", "total_chunks", "chunk_size"]
            for field in numeric_fields:
                if field in session_data:
                    try:
                        session_data[field] = int(session_data[field])
                    except (ValueError, TypeError):
                        logger.warning(f"Unable to convert {field} to int: {session_data.get(field)}")

            # Parse received_chunks JSON blob into a set
            if "received_chunks" in session_data:
                try:
                    received_chunks_data = session_data["received_chunks"]
                    if isinstance(received_chunks_data, str):
                        session_data["received_chunks"] = set(json.loads(received_chunks_data))
                    elif isinstance(received_chunks_data, list):
                        session_data["received_chunks"] = set(received_chunks_data)
                    else:
                        session_data["received_chunks"] = set()
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse received_chunks: {e}")
                    session_data["received_chunks"] = set()

            return session_data

        except Exception as e:
            logger.error(f"Failed to fetch session {upload_id}: {e}")
            return None

    async def update_session(self, upload_id: str, updates: Dict[str, Any]) -> bool:
        """Persist session updates."""
        try:
            # Prepare update payload
            update_data = updates.copy()
            update_data["updated_at"] = datetime.utcnow().isoformat()

            # Serialize received_chunks set/list as JSON
            if "received_chunks" in update_data:
                update_data["received_chunks"] = json.dumps(list(update_data["received_chunks"]))

            pipe = self.redis.pipeline()
            pipe.hset(self._get_session_key(upload_id), mapping=update_data)
            pipe.expire(self._get_session_key(upload_id), self.session_ttl)

            await pipe.execute()

            return True

        except Exception as e:
            logger.error(f"Failed to update session {upload_id}: {e}")
            return False

    async def add_chunk(self, upload_id: str, chunk_index: int) -> bool:
        """Record a received chunk index and update session."""
        try:
            # Fetch session to merge new chunk
            session = await self.get_session(upload_id)
            if not session:
                return False

            # Add chunk index
            received_chunks = session.get("received_chunks", set())
            received_chunks.add(chunk_index)

            # Persist updated set
            success = await self.update_session(upload_id, {
                "received_chunks": received_chunks,
                "chunk_" + str(chunk_index): datetime.utcnow().isoformat()
            })

            if success:
                # Set TTL for chunk metadata
                await self.redis.expire(self._get_chunk_key(upload_id, chunk_index), self.session_ttl)

            return success

        except Exception as e:
            logger.error(f"Failed to record chunk {upload_id}:{chunk_index}: {e}")
            return False

    async def has_chunk(self, upload_id: str, chunk_index: int) -> bool:
        """Return whether a chunk index was already received."""
        try:
            # Inspect session for chunk index
            session = await self.get_session(upload_id)
            if not session:
                return False

            received_chunks = session.get("received_chunks", set())
            return chunk_index in received_chunks

        except Exception as e:
            logger.error(f"Failed to check chunk {upload_id}:{chunk_index}: {e}")
            return False

    async def get_received_chunks(self, upload_id: str) -> Set[int]:
        """Return set of received chunk indices."""
        try:
            session = await self.get_session(upload_id)
            if not session:
                return set()

            return session.get("received_chunks", set())

        except Exception as e:
            logger.error(f"Failed to fetch received chunks {upload_id}: {e}")
            return set()

    async def delete_session(self, upload_id: str) -> bool:
        """Remove session metadata, indexes, chunk keys, and locks."""
        try:
            pipe = self.redis.pipeline()

            # Remove session hash
            pipe.delete(self._get_session_key(upload_id))

            # Remove from index
            pipe.zrem("upload:session:index", upload_id)

            # Delete chunk keys
            chunk_keys = []
            async for key in self.redis.scan_iter(match=f"{self.chunk_key_prefix}{upload_id}:*"):
                chunk_keys.append(key)

            if chunk_keys:
                pipe.delete(*chunk_keys)

            # Delete distributed lock key
            pipe.delete(self._get_lock_key(upload_id))

            await pipe.execute()

            logger.info(f"Session deleted: {upload_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {upload_id}: {e}")
            return False

    @asynccontextmanager
    async def acquire_lock(self, upload_id: str, timeout: Optional[int] = None):
        """Context manager implementing distributed lock for session operations."""
        request_id = str(uuid.uuid4())
        lock_key = self._get_lock_key(upload_id)
        lock_timeout = timeout or self.lock_timeout

        try:
            # Ensure timeout is an integer
            lock_timeout_int = int(lock_timeout)
            if lock_timeout_int <= 0:
                raise ValueError(f"Invalid lock timeout: {lock_timeout_int}")
            
            # Acquire lock via Lua script (KEYS[1]=lock_key, ARGV[1]=timeout, ARGV[2]=request_id)
            result = await self.redis.evalsha(
                self.lock_script_sha,
                1,
                lock_key,
                lock_timeout_int,
                request_id
            )

            if result:
                logger.debug(f"Redis session lock acquired: {upload_id}")
                try:
                    yield
                finally:
                    # Release lock via Lua
                    await self.redis.evalsha(
                        self.unlock_script_sha,
                        1,
                        lock_key,
                        request_id,
                        upload_id
                    )
                    logger.debug(f"Redis session lock released: {upload_id}")
            else:
                raise TimeoutError(f"Timed out acquiring session lock: {upload_id}")

        except Exception as e:
            logger.error(f"Session lock operation failed {upload_id}: {e}")
            raise

    async def cleanup_session(self, upload_id: str) -> bool:
        """Remove session metadata and associated temporary files."""
        try:
            async with self.acquire_lock(upload_id):
                # Fetch session info
                session = await self.get_session(upload_id)
                if not session:
                    return False

                # Remove temporary files
                import os

                from app.utils.file_utils import safe_remove

                temp_dir = session.get("temp_dir")
                if temp_dir and os.path.exists(temp_dir):
                    safe_remove(temp_dir)

                # Delete Redis structures
                await self.delete_session(upload_id)

                logger.info(f"Session cleanup completed: {upload_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to cleanup session {upload_id}: {e}")
            return False

    async def _cleanup_expired_sessions(self):
        """Background task that removes expired sessions."""
        logger.info("Upload session cleanup task started")

        while self._running:
            try:
                # Sleep one minute between scans
                await asyncio.sleep(60)

                # Locate expired sessions
                expired_sessions = await self.redis.zrangebyscore(
                    "upload:session:index",
                    0,
                    time.time() - self.session_ttl
                )

                if expired_sessions:
                    logger.info("Found %d expired sessions; cleaning up", len(expired_sessions))

                    for upload_id in expired_sessions:
                        try:
                            await self.cleanup_session(upload_id)
                        except Exception as e:
                            logger.error(f"Failed to cleanup expired session {upload_id}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup task error: {e}")
                await asyncio.sleep(10)  # retry after short delay

        logger.info("Upload session cleanup task stopped")

    async def get_session_stats(self) -> Dict[str, int]:
        """Return aggregate statistics about tracked sessions."""
        try:
            pipe = self.redis.pipeline()

            # Total sessions tracked
            pipe.zcard("upload:session:index")

            # Sessions created within the last hour
            one_hour_ago = time.time() - 3600
            pipe.zcount("upload:session:index", one_hour_ago, float('inf'))

            # Redis memory stats
            pipe.info("memory")

            results = await pipe.execute()

            return {
                "total_sessions": results[0],
                "active_sessions": results[1],
                "memory_used": results[2].get("used_memory_human", "N/A"),
                "connected_clients": results[2].get("connected_clients", 0)
            }

        except Exception as e:
            logger.error(f"Failed to fetch session stats: {e}")
            return {}

    async def recover_session(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Attempt to recover an existing session for resume purposes."""
        try:
            # Ensure session exists
            session = await self.get_session(upload_id)
            if not session:
                logger.warning(f"Attempted to recover non-existent session: {upload_id}")
                return None

            # Validate expiration
            expires_at = session.get("expires_at")
            if expires_at:
                expire_time = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                if datetime.utcnow() > expire_time:
                    logger.warning(f"Attempted to recover expired session: {upload_id}")
                    await self.delete_session(upload_id)
                    return None

            # Touch last_accessed to keep session active
            await self.update_session(upload_id, {"last_accessed": datetime.utcnow().isoformat()})

            logger.info(f"Session recovered successfully: {upload_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to recover session {upload_id}: {e}")
            return None


# Global Redis session manager instance
redis_session_manager = RedisSessionManager()


async def get_redis_session_manager() -> RedisSessionManager:
    """Return the global RedisSessionManager instance."""
    return redis_session_manager


# Convenience wrappers
async def create_session(
    upload_id: str,
    filename: str,
    total_size: int,
    total_chunks: int,
    chunk_size: int,
    temp_dir: str,
    temp_file: str,
    user_id: str = None
) -> bool:
    """Proxy helper for creating an upload session."""
    return await redis_session_manager.create_session(
        upload_id, filename, total_size, total_chunks, chunk_size,
        temp_dir, temp_file, user_id
    )


async def get_session(upload_id: str) -> Optional[Dict[str, Any]]:
    """Proxy helper for fetching a session."""
    return await redis_session_manager.get_session(upload_id)


async def update_session(upload_id: str, updates: Dict[str, Any]) -> bool:
    """Proxy helper for updating a session."""
    return await redis_session_manager.update_session(upload_id, updates)


async def add_chunk(upload_id: str, chunk_index: int) -> bool:
    """Proxy helper to record a received chunk."""
    return await redis_session_manager.add_chunk(upload_id, chunk_index)


async def delete_session(upload_id: str) -> bool:
    """Proxy helper that removes a session."""
    return await redis_session_manager.delete_session(upload_id)


async def cleanup_session(upload_id: str) -> bool:
    """Proxy helper that cleans up a session."""
    return await redis_session_manager.cleanup_session(upload_id)


async def recover_session(upload_id: str) -> Optional[Dict[str, Any]]:
    """Proxy helper that attempts to recover a session."""
    return await redis_session_manager.recover_session(upload_id)
