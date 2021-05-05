from hdbcli import dbapi
from hana_automl.utils.credentials import port, host, user, password
from hana_automl.storage import table_exists

"""For developers only. Cleanup database by deleting all test tables. Be careful!"""


def clean():
    SCHEMA = "DEVELOPER"
    conn = dbapi.connect(address=host, user=user, password=password, port=port)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM TABLES WHERE SCHEMA_NAME='{SCHEMA}'")
    res = cursor.fetchall()
    for string in res:
        if table_exists(cursor, SCHEMA, string[1]):
            cursor.execute(f'DROP TABLE "{string[1]}" CASCADE;')
    print(f"REMOVED ALL TABLES FROM {SCHEMA}")
