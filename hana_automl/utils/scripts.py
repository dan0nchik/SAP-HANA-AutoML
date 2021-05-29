from hana_ml import ConnectionContext


def setup_user(connection_context: ConnectionContext, username: str, create_new: bool, password: str = None):
    print('Checking that PAL is installed...')
    cursor = connection_context.connection.cursor()
    cursor.execute("""SELECT * FROM "SYS"."AFL_AREAS" WHERE AREA_NAME = 'AFLPAL';""")
    print(cursor.fetchall())
    cursor.execute("""SELECT * FROM "SYS"."AFL_PACKAGES" WHERE AREA_NAME = 'AFLPAL';""")
    print(cursor.fetchall())
    cursor.execute("""SELECT * FROM "SYS"."AFL_FUNCTIONS" WHERE AREA_NAME = 'AFLPAL';""")
    print(cursor.fetchall())
    if create_new:
        print('Creating new user...')
        cursor.execute(f'CREATE USER {username} PASSWORD {password};')
    print('Granting roles to user to execute PAL functions...')
    cursor.execute(f"""CALL GRANT_ACTIVATED_ROLE ('AFL__SYS_AFL_AFLPAL_EXECUTE','{username}')""")
    cursor.execute(f"""CALL GRANT_ACTIVATED_ROLE ('AFL__SYS_AFL_AFLPAL_EXECUTE_WITH_GRANT_OPTION','{username}')""")
    print('Done.')
