"""The tests to run in this project.
To run the tests type,
$ nosetests --verbose
"""

from nose.tools import assert_true
import requests

# BASE_URL = "http://127.0.0.1"
BASE_URL = "http://localhost:5000"
# BASE_URL = "https://python3-flask-uat.herokuapp.com/"


class NewUUID():  # pylint: disable=too-few-public-methods
    """The new uuid created when creating a new book request.
    The NewUUID is used for further tests
    """

    def __init__(self, value):
        self.value = value
    #NEW_UUID = None


def test_get_all_requests():
    "Test getting all requests"
    response = requests.get('%s/request' % (BASE_URL))
    assert_true(response.ok)


def test_get_individual_request():
    "Test getting an individual request"
    response = requests.get(
        '%s/request/04cfc704-acb2-40af-a8d3-4611fab54ada' % (BASE_URL))
    assert_true(response.ok)


def test_get_individual_request_404():
    "Test getting a non existent request"
    response = requests.get('%s/request/an_incorrect_id' % (BASE_URL))
    assert_true(response.status_code == 404)


def test_add_new_record():
    "Test adding a new record"
    payload = {'title': 'Good & Bad Book', 'email': 'testuser3@test.com'}
    response = requests.post('%s/request' % (BASE_URL), json=payload)
    NewUUID.value = str(response.json()['id'])
    assert_true(response.status_code == 201)


def test_get_new_record():
    "Test getting the new record"
    url = '%s/request/%s' % (BASE_URL, NewUUID.value)
    response = requests.get(url)
    assert_true(response.ok)


def test_edit_new_record_title():
    "Test editing the new records title"
    payload = {'title': 'edited Good & Bad Book',
               'email': 'testuser3@test.com'}
    response = requests.put('%s/request/%s' %
                            (BASE_URL, NewUUID.value), json=payload)
    assert_true(response.json()['title'] == "edited Good & Bad Book")


def test_edit_new_record_email():
    "Test editing the new records email"
    payload = {'title': 'edited Good & Bad Book',
               'email': 'testuser4@test.com'}
    response = requests.put('%s/request/%s' %
                            (BASE_URL, NewUUID.value), json=payload)
    assert_true(response.json()['email'] == "testuser4@test.com")


def test_add_new_record_bad_email_format():
    "Test adding a new record with a bad email"
    payload = {'title': 'Good & Bad Book', 'email': 'badEmailFormat'}
    response = requests.post('%s/request' % (BASE_URL), json=payload)
    assert_true(response.status_code == 400)


def test_add_new_record_bad_title_key():
    "Test adding a new record with a bad title"
    payload = {'badTitleKey': 'Good & Bad Book', 'email': 'testuser4@test.com'}
    response = requests.post('%s/request' % (BASE_URL), json=payload)
    assert_true(response.status_code == 400)


def test_add_new_record_no_email_key():
    "Test adding a new record no email"
    payload = {'title': 'Good & Bad Book'}
    response = requests.post('%s/request' % (BASE_URL), json=payload)
    assert_true(response.status_code == 400)


def test_add_new_record_no_title_key():
    "Test adding a new record no title"
    payload = {'email': 'testuser5@test.com'}
    response = requests.post('%s/request' % (BASE_URL), json=payload)
    assert_true(response.status_code == 400)


def test_add_new_record_unicode_title():
    "Test adding a new record with a unicode title"
    payload = {'title': '▚Ⓜ⌇⇲', 'email': 'testuser5@test.com'}
    response = requests.post('%s/request' % (BASE_URL), json=payload)
    assert_true(response.ok)


def test_add_new_record_no_payload():
    "Test adding a new record with no payload"
    payload = None
    response = requests.post('%s/request' % (BASE_URL), json=payload)
    assert_true(response.status_code == 400)


def test_delete_new_record():
    "Test deleting the new record"
    response = requests.delete('%s/request/%s' % (BASE_URL, NewUUID.value))
    assert_true(response.status_code == 204)


def test_delete_new_record_404():
    "Test deleting the new record that was already deleted"
    response = requests.delete('%s/request/%s' % (BASE_URL, NewUUID.value))
    assert_true(response.status_code == 404)
