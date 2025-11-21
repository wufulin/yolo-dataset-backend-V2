#!/usr/bin/env python3
"""检查 Redis 配置脚本"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import config_manager
from app.config import settings

def check_redis_config():
    """检查 Redis 配置"""
    print("=" * 60)
    print("Redis 配置检查")
    print("=" * 60)
    
    # 检查环境变量
    print("\n1. 环境变量检查:")
    redis_url_env = os.getenv("REDIS_URL")
    redis_host_env = os.getenv("REDIS_HOST")
    redis_port_env = os.getenv("REDIS_PORT")
    
    print(f"   REDIS_URL (环境变量): {redis_url_env or '未设置'}")
    print(f"   REDIS_HOST (环境变量): {redis_host_env or '未设置'}")
    print(f"   REDIS_PORT (环境变量): {redis_port_env or '未设置'}")
    
    # 检查 settings 中的值
    print("\n2. Settings 中的配置值:")
    print(f"   settings.redis_url: {settings.redis_url}")
    print(f"   settings.redis_host: {settings.redis_host}")
    print(f"   settings.redis_port: {settings.redis_port}")
    print(f"   settings.redis_db: {settings.redis_db}")
    print(f"   settings.redis_password: {'已设置' if settings.redis_password else '未设置'}")
    
    # 检查 .env.dev 文件
    print("\n3. .env.dev 文件检查:")
    env_file = project_root / ".env.dev"
    if env_file.exists():
        print(f"   ✓ .env.dev 文件存在: {env_file}")
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            redis_lines = [line.strip() for line in lines if line.strip().startswith('REDIS')]
            if redis_lines:
                print("   Redis 相关配置:")
                for line in redis_lines:
                    # 隐藏密码
                    if 'PASSWORD' in line and '=' in line:
                        key, value = line.split('=', 1)
                        if value.strip():
                            print(f"     {key}=***")
                        else:
                            print(f"     {line}")
                    else:
                        print(f"     {line}")
            else:
                print("   ⚠ 未找到 REDIS 相关配置")
    else:
        print(f"   ✗ .env.dev 文件不存在: {env_file}")
        print(f"   提示: 请从 env.example 复制并创建 .env.dev 文件")
    
    # 判断将使用哪种连接方式
    print("\n4. 连接方式判断:")
    redis_url_value = settings.redis_url.strip() if settings.redis_url else ""
    use_url = bool(redis_url_value and redis_url_value != "redis://localhost:6379/0")
    
    if use_url:
        print(f"   ✓ 将使用 Redis URL 连接: {redis_url_value}")
    else:
        print(f"   ✓ 将使用 host/port 连接: {settings.redis_host}:{settings.redis_port}")
        if not redis_url_value:
            print("   ⚠ REDIS_URL 未设置或为空，将使用 host/port 配置")
        elif redis_url_value == "redis://localhost:6379/0":
            print("   ⚠ REDIS_URL 是默认值，将使用 host/port 配置")
    
    # 配置建议
    print("\n5. 配置建议:")
    if not use_url and settings.redis_host != "localhost":
        print("   💡 建议使用 REDIS_URL 方式连接远程 Redis:")
        if settings.redis_password:
            print(f"      REDIS_URL=redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}")
        else:
            print(f"      REDIS_URL=redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        check_redis_config()
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

