from django.test import LiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


class LoginTest(LiveServerTestCase):
    def setUp(self):
        self.browser = webdriver.Chrome()
        self.browser.implicitly_wait(10)

    def tearDown(self):
        self.browser.quit()

    def test_user_can_login_with_valid_credentials(self):
        """Test that user can login with valid credentials and is redirected to landing page"""
        # Navigate to login page
        self.browser.get(self.live_server_url + '/')
        
        # Verify we're on the login page
        self.assertIn("Login to your account", self.browser.title)
        
        # Find form elements and fill them in
        username_field = self.browser.find_element(By.NAME, "username")
        password_field = self.browser.find_element(By.NAME, "password")
        
        # Enter valid credentials (Alex/1001 from the dummy users in login.html)
        username_field.send_keys("Alex")
        password_field.send_keys("1001")
        
        # Submit the form by clicking the login link
        login_link = self.browser.find_element(By.XPATH, "//a[@onclick='handleLogin(event)']")
        login_link.click()
        
        # Wait for redirect to landing page
        WebDriverWait(self.browser, 10).until(
            EC.url_contains("/selector/")
        )
        
        # Verify we're on the landing page
        current_url = self.browser.current_url
        self.assertIn("/selector/", current_url)
        
        # Wait for user session info to be populated (this shows login was successful)
        WebDriverWait(self.browser, 10).until(
            EC.presence_of_element_located((By.ID, "loggedInUserName"))
        )
        
        # Verify user session info is populated from localStorage
        user_name_element = self.browser.find_element(By.ID, "loggedInUserName")
        user_name_text = user_name_element.text
        
        # Check that the logged-in user name is displayed
        self.assertIn("Alex", user_name_text, "Logged in user name should be displayed")
        
        # Check that visit date is displayed
        visit_date_element = self.browser.find_element(By.ID, "loggedInUserVisitDate")
        visit_date_text = visit_date_element.text
        self.assertIn("07.07.2026", visit_date_text, "Last visit date should be displayed")
        
        # Verify we have successfully logged in by checking body content
        body = self.browser.find_element(By.TAG_NAME, "body").text
        self.assertIn("Alex", body, "User name should be in page content")
        self.assertIn("Last visit", body, "Visit info should be in page content")

    def test_user_cannot_login_with_invalid_credentials(self):
        """Test that user cannot login with invalid credentials"""
        # Navigate to login page
        self.browser.get(self.live_server_url + '/')
        
        # Find form elements and fill them with invalid credentials
        username_field = self.browser.find_element(By.NAME, "username")
        password_field = self.browser.find_element(By.NAME, "password")
        
        username_field.send_keys("invaliduser")
        password_field.send_keys("wrongpassword")
        
        # Submit the form
        login_link = self.browser.find_element(By.XPATH, "//a[@onclick='handleLogin(event)']")
        login_link.click()
        
        # Wait a moment for the error message to appear
        time.sleep(1)
        
        # Verify error message is displayed
        error_element = self.browser.find_element(By.ID, "loginError")
        self.assertTrue(error_element.is_displayed(), "Error message should be visible")
        self.assertIn("Invalid username or password", error_element.text)
        
        # Verify we're still on the login page
        current_url = self.browser.current_url
        self.assertNotIn("/selector/", current_url)

    def test_user_can_login_with_second_valid_credentials(self):
        """Test that user can login with second set of valid credentials (Alif/1002)"""
        # Navigate to login page
        self.browser.get(self.live_server_url + '/')
        
        # Find form elements and fill them in
        username_field = self.browser.find_element(By.NAME, "username")
        password_field = self.browser.find_element(By.NAME, "password")
        
        # Enter second set of valid credentials
        username_field.send_keys("Alif")
        password_field.send_keys("1002")
        
        # Submit the form
        login_link = self.browser.find_element(By.XPATH, "//a[@onclick='handleLogin(event)']")
        login_link.click()
        
        # Wait for redirect to landing page
        WebDriverWait(self.browser, 10).until(
            EC.url_contains("/selector/")
        )
        
        # Verify we're on the landing page
        current_url = self.browser.current_url
        self.assertIn("/selector/", current_url)

    def test_login_form_elements_exist(self):
        """Test that all required login form elements are present"""
        # Navigate to login page
        self.browser.get(self.live_server_url + '/')
        
        # Verify form elements exist
        username_field = self.browser.find_element(By.NAME, "username")
        password_field = self.browser.find_element(By.NAME, "password")
        ctype_select = self.browser.find_element(By.NAME, "ctype")
        
        # Verify element attributes
        self.assertEqual(username_field.get_attribute("type"), "text")
        self.assertEqual(password_field.get_attribute("type"), "password")
        
        # Verify select options
        options = ctype_select.find_elements(By.TAG_NAME, "option")
        option_values = [option.get_attribute("value") for option in options]
        self.assertIn("administrator", option_values)
        self.assertIn("researcher", option_values)
