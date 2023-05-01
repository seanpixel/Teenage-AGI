
from playwright.sync_api import sync_playwright
def find_and_click_by_attributes(page, attributes):
    selector = "button"
    for attr, value in attributes.items():
        selector += f'[{attr}="{value}"]'
    element = page.locator(selector)
    element.click()

def enter_zipcode_and_press_enter(page, zipcode):
    input_selector = 'input[data-test-id="FrontpageAddressQueryInput"]'
    element = page.locator(input_selector)
    element.fill(zipcode)

def run(playwright, zipcode:str, prompt:str):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    # Navigate to wolt.com
    page.goto("https://wolt.com")
    button_attributes = {
        "aria-disabled": "false",
        "role": "button",
        "type": "button",
        "data-localization-key": "gdpr-consents.banner.accept-button"
    }
    find_and_click_by_attributes(page, button_attributes)


    enter_zipcode_and_press_enter(page, zipcode)
    page.wait_for_load_state("networkidle")
    page.press('input[data-test-id="FrontpageAddressQueryInput"]', 'Enter')
    page.wait_for_load_state("networkidle")
    import time
    time.sleep(2)
    search_input_selector = '[data-test-id="VenuesOnlySearchInput"]'
    page.fill(search_input_selector, prompt)
    page.press(search_input_selector, 'Enter')
    time.sleep(2)

    resulting_url = page.url
    browser.close()

    return resulting_url

    def main(prompt:str):
        with sync_playwright() as playwright:
            run(playwright, zipcode, prompt)






