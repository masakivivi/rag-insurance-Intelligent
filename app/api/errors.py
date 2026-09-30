# coding: utf-8
"""异常体系（P0 基础版本，P3 扩展统一处理与健康检查联动）。"""


class InsuranceQAError(Exception):
    """保险问答系统异常基类。"""

    code = "INTERNAL_ERROR"
    status_code = 500


class ConfigError(InsuranceQAError):
    """配置错误（缺 API key 等）。"""

    code = "CONFIG_ERROR"
    status_code = 500


class IndexError_(InsuranceQAError):
    """索引加载/同步失败。"""

    code = "INDEX_ERROR"
    status_code = 503


class SecurityError(InsuranceQAError):
    """安全校验失败（注入命中、输入非法等）。"""

    code = "SECURITY_ERROR"
    status_code = 400


class LLMError(InsuranceQAError):
    """LLM 调用失败（额度耗尽、超时等）。"""

    code = "LLM_ERROR"
    status_code = 502
