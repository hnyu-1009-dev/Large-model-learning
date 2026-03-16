# LangGraph Learning Guide

这套学习代码按照文档章节重新整理，目标不是“照抄文档示例”，而是把概念拆成一组可以逐个阅读、逐个运行、逐个验证的脚本。

建议学习顺序：

1. `01_hello_langgraph.py`
   覆盖：LangGraph 是什么、StateGraph 基本工作方式、节点/边/状态、编译与调用、可视化。
2. `02_state_schema.py`
   覆盖：`state_schema`、`input_schema`、`output_schema`、私有状态传递、`dataclass` 默认值。
3. `03_state_default_reducer.py`
   覆盖：Reducer、默认覆盖、`operator.add`、`operator.mul`、自定义 reducer、`add_messages`。
4. `04_nodes_edges_send_command_runtime.py`
   覆盖：`START` / `END`、普通边、条件边、条件入口、循环、`Send`、`Command`、`runtime.context`。
5. `05_advanced_execution.py`
   覆盖：节点缓存、重试策略、延迟节点执行、异步节点。
6. `06_persistence_streaming_interrupts.py`
   覆盖：线程、检查点、内存/SQLite 持久化、流式输出、自定义流、中断与恢复。
7. `07_memory_and_time_travel.py`
   覆盖：短期记忆、长期记忆、时间旅行、消息裁剪、删除、总结。
8. `08_subgraphs.py`
   覆盖：子图作为节点、节点中调用子图、父图导航、查看子图状态、流式输出子图。
9. `09_functional_api.py`
   覆盖：`@entrypoint`、`@task`、并行执行、恢复、`entrypoint.final`、混合调用 Graph API。
10. `10_state_chatbot_example.py`
    覆盖：把前面的概念组合成一个“有短期记忆 + 长期记忆 + 人工审批”的迷你聊天机器人。

环境准备：

```bash
python -m pip install -U langgraph langgraph-checkpoint-sqlite grandalf
```

阅读建议：

- 每个脚本都先读文件顶部注释，再看 `main()`。
- 每个演示函数都尽量只讲一个核心概念，方便你逐段运行。
- 代码里大量使用“为什么这样写”的注释，重点理解“状态如何流动”和“控制流如何跳转”。

运行建议：

```bash
python 01_hello_langgraph.py
python 02_state_schema.py
...
python 10_state_chatbot_example.py
```
