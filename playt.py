import asyncio
import pprint
from typing import TYPE_CHECKING, Optional

from langchain import agents
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_community.chat_models import tongyi
from langchain_community.tools.playwright import (
    ClickTool,
    CurrentWebPageTool,
    ExtractHyperlinksTool,
    ExtractTextTool,
    GetElementsTool,
    NavigateBackTool,
    NavigateTool,
    utils
)
from playwright.async_api import async_playwright

from loggers.screen_logger import log_agent_response, log_playwright_tool_call
from tools.fill_text_tool import FillTextTool

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.sync_api import Browser as SyncBrowser

async def main():
    """
    主函数
    """
    # 启动 Playwright
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
 

    @tool
    async def fill_text(browser: Optional["AsyncBrowser"], selector: str, text: str) -> str:

        """Fill text into an input field identified by the selector."""
        page = await utils.aget_current_page(browser)
        await page.fill(selector, text)
        return f"Filled text '{text}' into element with selector '{selector}'"
    try:
        # 创建工具（推荐组合使用）
        tools = [
            FillTextTool(async_browser=browser),
            ClickTool(async_browser=browser, playwright_timeout=10000,visible_only=False),
            CurrentWebPageTool(async_browser=browser),
            ExtractTextTool(async_browser=browser),
            NavigateTool(async_browser=browser),
            NavigateBackTool(async_browser=browser),
            GetElementsTool(async_browser=browser),
            ExtractHyperlinksTool(async_browser=browser),
        ]

        # 创建模型（注意：用 model_name）
        qwen_model = tongyi.ChatTongyi(
            model_name="qwen3-vl-plus", 
            temperature=0,
        )

        # 创建 agent
        browser_agent = agents.create_agent(
            model=qwen_model, 
            tools=tools,
            middleware=[
                log_playwright_tool_call,
                log_agent_response,
            ],
            ) 
        inputs={
            "messages": [
                HumanMessage(
                    #content="你是一个网页助手。当你需要填写表单时，必须一次只执行一个动作,不要在同一步中调用多个 fill_text 工具。打开 https://www.saucedemo.com/，然后输入账号:standard_user，密码：secret_sauce，然后点击登录，之后告诉我页面中有什么",
                    #content="你是一个网页助手。打开 https://www.saucedemo.com/，之后告诉我页面中有什么"
                    content="你是一个网页助手。打开http://zhjw.qfnu.edu.cn/jsxsd/framework/xsMain.jsp，登录账号：'2023413470'，密码：'Wang123.',自动识别并输入验证吗,然后告诉我页面中有什么内容"
                )
            ]
        }
        async for chunk in browser_agent.astream(inputs, stream_mode="updates"):
            pprint.pprint(chunk)
            print("\n"+"="*50+"\n")
    finally:
        # 清理资源
        await browser.close()
        await p.stop()


if __name__ == "__main__":
    asyncio.run(main())
