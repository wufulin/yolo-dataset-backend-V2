"""Redis服务 - 提供Redis客户端和会话管理功能"""
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
    """Redis会话管理器"""

    def __init__(self):
        """初始化Redis客户端和连接池"""
        self.redis: Optional[Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # 会话相关配置
        self.session_ttl = settings.redis_session_ttl
        self.lock_timeout = settings.redis_session_lock_timeout
        self.session_key_prefix = "upload:session:"
        self.chunk_key_prefix = "upload:chunk:"
        self.lock_key_prefix = "upload:lock:"
        self.lock_script_sha: Optional[str] = None

    async def connect(self) -> None:
        """连接Redis（单机配置）"""
        try:
            # 优先使用 redis_url（如果提供且不是默认值），否则使用 host/port
            redis_url_value = getattr(settings, 'redis_url', None) or ""
            redis_url_value = redis_url_value.strip() if redis_url_value else ""
            
            # 判断是否使用 URL：如果设置了 redis_url 且不是默认的 localhost:6379/0，则使用 URL
            # 注意：即使是 localhost，如果用户明确设置了 REDIS_URL，也应该使用它
            use_url = bool(redis_url_value and redis_url_value != "redis://localhost:6379/0")
            
            # 调试日志：显示实际读取的配置（INFO 级别，方便排查问题）
            logger.info(f"Redis配置检查 - redis_url: {redis_url_value[:80] if redis_url_value else '未设置（使用默认值）'}, "
                        f"redis_host: {settings.redis_host}, redis_port: {settings.redis_port}, "
                        f"使用URL连接: {use_url}")
            
            if use_url:
                # 使用 URL 连接（单机 Redis）
                # 从 URL 中提取信息用于日志（隐藏密码）
                url_for_log = redis_url_value.split('@')[-1] if '@' in redis_url_value else redis_url_value
                logger.info(f"使用 Redis URL 连接: {url_for_log}")
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
                # 使用 host/port 连接（单机 Redis）
                logger.info(f"使用 Redis host/port 连接: {settings.redis_host}:{settings.redis_port}")
                # 创建连接池
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

                # 创建Redis客户端
                self.redis = redis.Redis(
                    connection_pool=self._connection_pool,
                    encoding="utf-8",
                    decode_responses=True,
                )

            # 测试连接
            await self.redis.ping()
            
            # 验证可写性（单机 Redis 应该总是可写的）
            test_key = f"__test_write_{int(time.time())}"
            try:
                await self.redis.set(test_key, "test", ex=1)  # 1秒后过期
                await self.redis.delete(test_key)
                logger.info("Redis连接成功（单机模式，可写）")
            except ResponseError as e:
                error_msg = str(e).lower()
                if "read only" in error_msg or "readonly" in error_msg:
                    error_msg_full = (
                        "Redis连接失败：连接到只读副本，无法写入。\n"
                        "单机 Redis 配置应该总是可写的。\n"
                        "请检查：\n"
                        f"  1. Redis 配置是否正确: redis_url={redis_url_value if use_url else 'N/A'}, "
                        f"host={settings.redis_host}, port={settings.redis_port}\n"
                        "  2. Redis 服务器是否正常运行\n"
                        "  3. 是否意外连接到了只读副本\n"
                        "  4. Redis 服务器配置是否启用了只读模式（检查 redis.conf 中的 'replica-read-only' 或 'slave-read-only' 配置）\n"
                        "  5. 如果使用 Redis 集群，确保连接到主节点而不是副本节点"
                    )
                    logger.error(error_msg_full)
                    # 清理已创建的连接
                    if self.redis:
                        await self.redis.close()
                    if self._connection_pool:
                        await self._connection_pool.disconnect()
                    raise ConnectionError(error_msg_full)
                # 其他错误直接抛出
                logger.warning(f"Redis 写入测试失败（但连接正常）: {e}")
                raise

            # 预加载Lua脚本
            await self._load_lua_scripts()

            # 启动后台清理任务
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            # 清理资源
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
        """断开Redis连接"""
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

        logger.info("Redis连接已断开")

    async def _load_lua_scripts(self):
        """加载Lua脚本"""
        try:
            # 会话分布式锁脚本
            lock_script = """
            -- KEYS[1]: 锁键名
            -- ARGV[1]: 锁超时时间(秒) - 必须是整数
            -- ARGV[2]: 请求ID(客户端唯一标识)

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

            # 解锁脚本
            unlock_script = """
            -- KEYS[1]: 锁键名
            -- ARGV[1]: 请求ID
            -- ARGV[2]: 会话ID

            local lock_value = redis.call("GET", KEYS[1])
            if lock_value == ARGV[1] then
                redis.call("DEL", KEYS[1])
                -- 删除会话状态
                redis.call("DEL", "upload:session:" .. ARGV[2])
                return 1
            else
                return 0
            end
            """

            # 注册脚本
            self.lock_script_sha = await self.redis.script_load(lock_script)
            self.unlock_script_sha = await self.redis.script_load(unlock_script)

            logger.info("Redis Lua脚本加载成功")

        except Exception as e:
            logger.error(f"Lua脚本加载失败: {e}")
            raise

    def _get_session_key(self, upload_id: str) -> str:
        """获取会话键名"""
        return f"{self.session_key_prefix}{upload_id}"

    def _get_chunk_key(self, upload_id: str, chunk_index: int) -> str:
        """获取分片键名"""
        return f"{self.chunk_key_prefix}{upload_id}:{chunk_index}"

    def _get_lock_key(self, upload_id: str) -> str:
        """获取锁键名"""
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
        """创建上传会话"""
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

            # 使用管道提高性能
            pipe = self.redis.pipeline()
            pipe.hset(self._get_session_key(upload_id), mapping=session_data)
            pipe.expire(self._get_session_key(upload_id), self.session_ttl)
            pipe.zadd("upload:session:index", {upload_id: time.time()})
            pipe.expire("upload:session:index", self.session_ttl)

            await pipe.execute()

            logger.info(f"创建上传会话: {upload_id}")
            return True

        except Exception as e:
            logger.error(f"创建会话失败 {upload_id}: {e}")
            return False

    async def get_session(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """获取上传会话"""
        try:
            session_data_raw = await self.redis.hgetall(self._get_session_key(upload_id))

            if not session_data_raw:
                return None

            # 转换字节类型的键和值为字符串（即使设置了 decode_responses=True，hgetall 有时仍返回字节）
            session_data = {}
            for key, value in session_data_raw.items():
                # 转换键：如果是字节类型，解码为字符串
                if isinstance(key, bytes):
                    key_str = key.decode('utf-8')
                else:
                    key_str = str(key)
                
                # 转换值：如果是字节类型，解码为字符串
                if isinstance(value, bytes):
                    value_str = value.decode('utf-8')
                else:
                    value_str = str(value) if value is not None else None
                
                session_data[key_str] = value_str

            # 转换数值字段为整数
            numeric_fields = ["total_size", "total_chunks", "chunk_size"]
            for field in numeric_fields:
                if field in session_data:
                    try:
                        session_data[field] = int(session_data[field])
                    except (ValueError, TypeError):
                        logger.warning(f"无法转换 {field} 为整数: {session_data.get(field)}")

            # 转换received_chunks从JSON字符串
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
                    logger.warning(f"无法解析 received_chunks: {e}")
                    session_data["received_chunks"] = set()

            return session_data

        except Exception as e:
            logger.error(f"获取会话失败 {upload_id}: {e}")
            return None

    async def update_session(self, upload_id: str, updates: Dict[str, Any]) -> bool:
        """更新会话"""
        try:
            # 准备更新数据
            update_data = updates.copy()
            update_data["updated_at"] = datetime.utcnow().isoformat()

            # 转换received_chunks为JSON
            if "received_chunks" in update_data:
                update_data["received_chunks"] = json.dumps(list(update_data["received_chunks"]))

            pipe = self.redis.pipeline()
            pipe.hset(self._get_session_key(upload_id), mapping=update_data)
            pipe.expire(self._get_session_key(upload_id), self.session_ttl)

            await pipe.execute()

            return True

        except Exception as e:
            logger.error(f"更新会话失败 {upload_id}: {e}")
            return False

    async def add_chunk(self, upload_id: str, chunk_index: int) -> bool:
        """添加已接收的分片"""
        try:
            # 获取当前会话
            session = await self.get_session(upload_id)
            if not session:
                return False

            # 添加分片索引
            received_chunks = session.get("received_chunks", set())
            received_chunks.add(chunk_index)

            # 更新会话
            success = await self.update_session(upload_id, {
                "received_chunks": received_chunks,
                "chunk_" + str(chunk_index): datetime.utcnow().isoformat()
            })

            if success:
                # 设置分片临时过期时间(比如1小时)
                await self.redis.expire(self._get_chunk_key(upload_id, chunk_index), self.session_ttl)

            return success

        except Exception as e:
            logger.error(f"添加分片失败 {upload_id}:{chunk_index}: {e}")
            return False

    async def has_chunk(self, upload_id: str, chunk_index: int) -> bool:
        """检查分片是否存在"""
        try:
            # 检查会话中是否有此分片
            session = await self.get_session(upload_id)
            if not session:
                return False

            received_chunks = session.get("received_chunks", set())
            return chunk_index in received_chunks

        except Exception as e:
            logger.error(f"检查分片失败 {upload_id}:{chunk_index}: {e}")
            return False

    async def get_received_chunks(self, upload_id: str) -> Set[int]:
        """获取已接收的分片"""
        try:
            session = await self.get_session(upload_id)
            if not session:
                return set()

            return session.get("received_chunks", set())

        except Exception as e:
            logger.error(f"获取已接收分片失败 {upload_id}: {e}")
            return set()

    async def delete_session(self, upload_id: str) -> bool:
        """删除会话"""
        try:
            pipe = self.redis.pipeline()

            # 删除会话数据
            pipe.delete(self._get_session_key(upload_id))

            # 从索引中删除
            pipe.zrem("upload:session:index", upload_id)

            # 删除所有相关分片
            chunk_keys = []
            async for key in self.redis.scan_iter(match=f"{self.chunk_key_prefix}{upload_id}:*"):
                chunk_keys.append(key)

            if chunk_keys:
                pipe.delete(*chunk_keys)

            # 删除锁
            pipe.delete(self._get_lock_key(upload_id))

            await pipe.execute()

            logger.info(f"删除会话: {upload_id}")
            return True

        except Exception as e:
            logger.error(f"删除会话失败 {upload_id}: {e}")
            return False

    @asynccontextmanager
    async def acquire_lock(self, upload_id: str, timeout: Optional[int] = None):
        """获取分布式会话锁"""
        request_id = str(uuid.uuid4())
        lock_key = self._get_lock_key(upload_id)
        lock_timeout = timeout or self.lock_timeout

        try:
            # 确保 lock_timeout 是整数
            lock_timeout_int = int(lock_timeout)
            if lock_timeout_int <= 0:
                raise ValueError(f"Invalid lock timeout: {lock_timeout_int}")
            
            # 使用Lua脚本获取锁
            # 参数顺序：KEYS[1]=lock_key, ARGV[1]=lock_timeout, ARGV[2]=request_id
            result = await self.redis.evalsha(
                self.lock_script_sha,
                1,  # KEYS 数量
                lock_key,  # KEYS[1]
                lock_timeout_int,  # ARGV[1] - 锁超时时间（秒）
                request_id  # ARGV[2] - 请求ID
            )

            if result:
                logger.debug(f"获取会话锁成功: {upload_id}")
                try:
                    yield
                finally:
                    # 释放锁
                    await self.redis.evalsha(
                        self.unlock_script_sha,
                        1,
                        lock_key,
                        request_id,
                        upload_id
                    )
                    logger.debug(f"释放会话锁成功: {upload_id}")
            else:
                raise TimeoutError(f"无法获取会话锁: {upload_id}")

        except Exception as e:
            logger.error(f"会话锁操作失败 {upload_id}: {e}")
            raise

    async def cleanup_session(self, upload_id: str) -> bool:
        """清理会话及相关资源"""
        try:
            async with self.acquire_lock(upload_id):
                # 获取会话信息
                session = await self.get_session(upload_id)
                if not session:
                    return False

                # 删除所有分片文件
                import os

                from app.utils.file_utils import safe_remove

                temp_dir = session.get("temp_dir")
                if temp_dir and os.path.exists(temp_dir):
                    safe_remove(temp_dir)

                # 删除Redis中的会话数据
                await self.delete_session(upload_id)

                logger.info(f"清理会话完成: {upload_id}")
                return True

        except Exception as e:
            logger.error(f"清理会话失败 {upload_id}: {e}")
            return False

    async def _cleanup_expired_sessions(self):
        """清理过期会话的后台任务"""
        logger.info("启动会话清理任务")

        while self._running:
            try:
                # 每分钟检查一次
                await asyncio.sleep(60)

                # 查找过期的会话
                expired_sessions = await self.redis.zrangebyscore(
                    "upload:session:index",
                    0,
                    time.time() - self.session_ttl
                )

                if expired_sessions:
                    logger.info(f"发现{len(expired_sessions)}个过期会话，开始清理")

                    for upload_id in expired_sessions:
                        try:
                            await self.cleanup_session(upload_id)
                        except Exception as e:
                            logger.error(f"清理过期会话失败 {upload_id}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"会话清理任务异常: {e}")
                await asyncio.sleep(10)  # 出错时稍后重试

        logger.info("会话清理任务已停止")

    async def get_session_stats(self) -> Dict[str, int]:
        """获取会话统计信息"""
        try:
            pipe = self.redis.pipeline()

            # 总会话数
            pipe.zcard("upload:session:index")

            # 活跃会话数(最近1小时内创建的)
            one_hour_ago = time.time() - 3600
            pipe.zcount("upload:session:index", one_hour_ago, float('inf'))

            # 内存使用情况
            pipe.info("memory")

            results = await pipe.execute()

            return {
                "total_sessions": results[0],
                "active_sessions": results[1],
                "memory_used": results[2].get("used_memory_human", "N/A"),
                "connected_clients": results[2].get("connected_clients", 0)
            }

        except Exception as e:
            logger.error(f"获取会话统计失败: {e}")
            return {}

    async def recover_session(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """恢复会话"""
        try:
            # 检查会话是否存在
            session = await self.get_session(upload_id)
            if not session:
                logger.warning(f"尝试恢复不存在的会话: {upload_id}")
                return None

            # 检查会话是否过期
            expires_at = session.get("expires_at")
            if expires_at:
                expire_time = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                if datetime.utcnow() > expire_time:
                    logger.warning(f"尝试恢复过期的会话: {upload_id}")
                    await self.delete_session(upload_id)
                    return None

            # 更新会话最后访问时间
            await self.update_session(upload_id, {"last_accessed": datetime.utcnow().isoformat()})

            logger.info(f"会话恢复成功: {upload_id}")
            return session

        except Exception as e:
            logger.error(f"恢复会话失败 {upload_id}: {e}")
            return None


# 全局Redis会话管理器实例
redis_session_manager = RedisSessionManager()


async def get_redis_session_manager() -> RedisSessionManager:
    """获取Redis会话管理器实例"""
    return redis_session_manager


# 便捷方法
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
    """创建上传会话"""
    return await redis_session_manager.create_session(
        upload_id, filename, total_size, total_chunks, chunk_size,
        temp_dir, temp_file, user_id
    )


async def get_session(upload_id: str) -> Optional[Dict[str, Any]]:
    """获取上传会话"""
    return await redis_session_manager.get_session(upload_id)


async def update_session(upload_id: str, updates: Dict[str, Any]) -> bool:
    """更新会话"""
    return await redis_session_manager.update_session(upload_id, updates)


async def add_chunk(upload_id: str, chunk_index: int) -> bool:
    """添加已接收的分片"""
    return await redis_session_manager.add_chunk(upload_id, chunk_index)


async def delete_session(upload_id: str) -> bool:
    """删除会话"""
    return await redis_session_manager.delete_session(upload_id)


async def cleanup_session(upload_id: str) -> bool:
    """清理会话"""
    return await redis_session_manager.cleanup_session(upload_id)


async def recover_session(upload_id: str) -> Optional[Dict[str, Any]]:
    """恢复会话"""
    return await redis_session_manager.recover_session(upload_id)
