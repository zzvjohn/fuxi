"""
量化数据平台凭据加载模块
==========================
从 .env 文件读取 Tushare / 聚宽 的 API 凭据，
提供统一的初始化接口供各 Agent 调用。

使用方式：
    from credentials import get_tushare_api, get_jqdata_auth

    # Tushare
    pro = get_tushare_api()

    # 聚宽
    get_jqdata_auth()
"""

import configparser
import os

# 配置文件路径
ENV_PATH = os.path.join(os.path.dirname(__file__), '.env')


def _load_config():
    """读取 .env 配置文件"""
    config = configparser.ConfigParser()
    if not os.path.exists(ENV_PATH):
        raise FileNotFoundError(
            f"凭据文件不存在: {ENV_PATH}\n"
            "请参照 .env.example 创建配置文件"
        )
    config.read(ENV_PATH, encoding='utf-8')
    return config


def get_tushare_api():
    """
    初始化并返回 Tushare Pro API 实例
    
    Returns:
        tushare.pro_api 实例
    
    Usage:
        import tushare as ts
        pro = get_tushare_api()
        df = pro.daily(ts_code='600519.SH', start_date='20250101')
    """
    import tushare as ts
    
    config = _load_config()
    token = config.get('tushare', 'token')
    
    # 直接传 token 初始化, 避免权限问题 (不写 tk.csv)
    pro = ts.pro_api(token)
    print("[OK] Tushare Pro connected")
    return pro


def get_tushare_token():
    """返回 Tushare token 字符串（用于自定义超时等场景）"""
    config = _load_config()
    return config.get('tushare', 'token')


def get_jqdata_auth():
    """
    认证聚宽 JQData SDK
    
    Usage:
        from jqdatasdk import auth
        get_jqdata_auth()  # 认证
        # 之后正常使用 jqdatasdk 的 get_price 等函数
    """
    from jqdatasdk import auth
    
    config = _load_config()
    username = config.get('joinquant', 'username')
    password = config.get('joinquant', 'password')
    
    auth(username, password)
    print("[✓] 聚宽 JQData 已认证")


def get_wqbrain_auth():
    """
    获取 WorldQuant BRAIN 凭据
    
    Returns:
        tuple: (email, password)
    """
    config = _load_config()
    email = config.get('worldquant', 'email')
    password = config.get('worldquant', 'password')
    return email, password


def get_credentials_info():
    """
    获取凭据摘要（脱敏，仅用于确认配置是否正确）
    
    Returns:
        dict: 各平台的连接状态和脱敏信息
    """
    config = _load_config()
    
    token = config.get('tushare', 'token')
    username = config.get('joinquant', 'username')
    
    return {
        'tushare_token': f"{token[:6]}...{token[-6:]}" if len(token) > 12 else "***",
        'jqdata_username': username[:3] + "****" + username[-2:],
        'env_path': ENV_PATH,
    }


if __name__ == '__main__':
    # 测试凭据配置是否正确
    info = get_credentials_info()
    print("=== 凭据配置检查 ===")
    print(f"Tushare Token: {info['tushare_token']}")
    print(f"聚宽账号:      {info['jqdata_username']}")
    print(f"配置文件:      {info['env_path']}")
    print()
    
    # 尝试连接 Tushare
    try:
        pro = get_tushare_api()
        print("[✓] Tushare 连接成功")
    except Exception as e:
        print(f"[✗] Tushare 连接失败: {e}")
    
    # 尝试连接聚宽
    try:
        get_jqdata_auth()
        print("[✓] 聚宽 JQData 连接成功")
    except Exception as e:
        print(f"[✗] 聚宽 JQData 连接失败: {e}")
