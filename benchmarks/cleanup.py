from hana_ml import ConnectionContext

from hana_automl.storage import table_exists

"""For developers only. Cleanup database by deleting all test tables. Be careful!"""


def clean(cc: ConnectionContext, schema: str):
    cursor = cc.connection.cursor()
    cursor.execute(f"SELECT * FROM TABLES WHERE SCHEMA_NAME='{schema}'")
    res = cursor.fetchall()
    for string in res:
        if table_exists(cursor, schema, string[1]):
            cursor.execute(f'DROP TABLE "{string[1]}" CASCADE;')
    print(f"REMOVED ALL TABLES FROM {schema}")
