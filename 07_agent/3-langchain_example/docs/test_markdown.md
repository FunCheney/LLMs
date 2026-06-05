# 测试Markdown文档

这是一个用于测试多模态文档处理功能的Markdown文件。

## 图片示例

下面是一些图片引用的示例：

![LangChain Logo](https://python.langchain.com/img/brand/wordmark.png)

![OpenAI Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/512px-OpenAI_Logo.svg.png)

## 本地图片引用

如果有本地图片文件，可以这样引用：

![本地图片](./images/sample.png)

![另一个本地图片](../assets/logo.jpg)

## 文档内容

这个文档用于演示：
1. 图片引用的识别和提取
2. URL替换功能
3. 多模态文档处理流程

处理后，所有的图片引用都会被替换为COS存储的URL。