from django.test import TestCase, SimpleTestCase

class SimpleTest(SimpleTestCase):
    def test_home_case_status(self):
        response = self.client.get('/')
        self.assertEquals(response.status_code, 200)


# Create your tests here.
