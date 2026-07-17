# 外部调用可靠性

ai-clip 对外部调用采用“显式允许、有限次数”的重试策略。没有声明安全边界的调用不自动
重试，尤其不能盲目重放异步生成任务或可能已经产生费用的请求。

## 失败分类

共享分类定义在 `core/retry.py`：

| category | 示例 | 默认处理 |
|----------|------|----------|
| `configuration` | 缺 API key | 立即失败 |
| `authentication` | HTTP 401/403 | 立即失败 |
| `rate_limit` | HTTP 429 | 按服务策略重试 |
| `timeout` | connect/pool timeout、HTTP 408/504 | 按服务策略重试 |
| `transient` | connect error、HTTP 5xx | 按服务策略重试 |
| `invalid_response` | 非法 JSON、缺少必要字段 | 立即失败 |
| `terminal` | 其他 4xx、非网络异常 | 立即失败 |

终止错误包含 `service / operation / category / attempts / status`，不包含响应正文、请求 header
或密钥。现有 workflow tracker 会把该摘要写入 run-status 的阶段 `error`。

## 当前策略

| 服务 | 默认总 attempts | 自动重试 | 不自动重试 |
|------|----------------|----------|------------|
| 主 LLM / Pair Review | 3 | 429、408/504、5xx、transport timeout/error | 认证、其他 4xx、非法响应 |
| Tavily | 2/每个 query | 429、408/504、5xx、transport timeout/error | 认证、其他 4xx、非法响应 |
| MiMo TTS | 2 | 429、408/504、5xx、connect/connect-timeout/pool-timeout | read/write/protocol error、认证、非法响应 |
| yt-dlp | 由 yt-dlp 配置管理 | 当前 `retries=1`、`extractor_retries=1` | 由采集诊断隔离 |
| ComfyUI | 1 次提交 | 预存 caller-assigned prompt id，恢复时继续轮询 | POST 不盲目重放 |
| MoneyPrinter | 1 次提交 | 获得 server task id 后持久化并恢复轮询 | task id 未知时禁止自动重提 |

MiMo 的 read timeout 特意不重试：服务端可能已经完成生成并计费，只是响应在返回途中断开。
连接尚未建立、服务明确返回 429/5xx 时才允许有限重放。

退避为 1 秒、2 秒、最多 4 秒。可在 YAML 中调整总 attempts：

```yaml
llm:
  max_attempts: 3
pair:
  max_attempts: 3
source_research:
  max_attempts: 2
tts:
  max_attempts: 2
```

`source_research.max_searches` 控制一次 research 使用几个不同角度的 query；`max_attempts` 只控制
同一 query 遇到临时网络错误时最多发送几次，两者不能混用。

## 异步任务恢复

ComfyUI 图片和 MoneyPrinter 视频在输出文件旁写入 `<output>.job.json`，记录请求指纹、远端
任务 ID 和 `submitting / submitted / running / succeeded / failed / unknown` 状态。相同请求再次
运行时优先恢复既有任务，不再提交第二份；不同请求不能覆盖仍活动的任务。

ComfyUI 支持客户端预先指定 UUID，所以 ai-clip 在 POST 前即可保存 `prompt_id`。即使提交响应
丢失，也能用同一个 ID 继续查询 history。明确连接失败或远端执行失败会记为 `failed`；超时、
轮询中断和暂时无法确认的响应保留为可恢复状态。

MoneyPrinter 的 `task_id` 由服务端在接受请求后生成。如果响应已返回，后续中断可从 sidecar
恢复轮询和下载；如果响应丢失且尚未获得 ID，状态记为 `unknown`，后续运行会拒绝自动重提。
此时应先在 MoneyPrinter 服务端确认是否已有任务，确认没有后再删除对应 `.job.json` 重新运行。
直接删除未确认的 `unknown` 状态可能生成重复任务。

输出采用临时文件后原子替换。ComfyUI 图片已经写入、但资产 manifest 尚未来得及写入时，
assets 阶段会依据 job sidecar 完成 manifest；清理系统生成的孤儿图片时也会同时清理 sidecar。

## 使用量

成功调用仍按一个逻辑 call 计入 `cost.jsonl`，并额外记录：

- `attempts`：该逻辑调用实际发送次数。
- `retries`：`attempts - 1`，在 run usage 的 total 中汇总。

这不会假装知道失败尝试是否产生账单；供应商最终账单仍是费用权威来源。
