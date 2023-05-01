
from playwright.async_api import async_playwright, Playwright

async def find_and_click_by_attributes(page, attributes):
    selector = "button"
    for attr, value in attributes.items():
        selector += f'[{attr}="{value}"]'
    element = page.locator(selector)
    await element.click()

async def enter_zipcode_and_press_enter(page, zipcode):
    input_selector = 'input[data-test-id="FrontpageAddressQueryInput"]'
    element = page.locator(input_selector)
    await element.fill(zipcode)
    await element.press("Enter")

async def run(playwright, zipcode:str, prompt:str):
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()

    # Navigate to wolt.com
    await page.goto("https://wolt.com")
    button_attributes = {
        "aria-disabled": "false",
        "role": "button",
        "type": "button",
        "data-localization-key": "gdpr-consents.banner.accept-button"
    }
    await find_and_click_by_attributes(page, button_attributes)

    await enter_zipcode_and_press_enter(page, zipcode)
    await page.wait_for_load_state("networkidle")
    await page.press('input[data-test-id="FrontpageAddressQueryInput"]', 'Enter')
    await page.wait_for_load_state("networkidle")
    await page.wait_for_selector('[data-test-id="VenuesOnlySearchInput"]')
    await page.wait_for_load_state("networkidle")
    search_input_selector = '[data-test-id="VenuesOnlySearchInput"]'
    await page.wait_for_load_state("networkidle")
    element = page.locator(search_input_selector)
    await element.fill(prompt)
    await page.press('input[data-test-id="VenuesOnlySearchInput"]', 'Enter')
    await page.wait_for_load_state("networkidle")

    resulting_url = page.url
    await browser.close()

    return resulting_url

async def main(prompt:str, zipcode:str):
    async with async_playwright() as playwright:
        result = await run(playwright, zipcode, prompt)
        print(result)
        return result
import asyncio
# asyncio.run(main(prompt="pizza", zipcode="10005"))