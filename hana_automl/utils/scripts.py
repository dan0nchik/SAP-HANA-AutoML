from hana_ml import ConnectionContext


def setup_user(connection_context: ConnectionContext, username: str, password: str):
    print("Checking that PAL is installed...")
    cursor = connection_context.connection.cursor()
    print(
        "AFL_AREAS",
        connection_context.sql(
            """SELECT * FROM "SYS"."AFL_AREAS" WHERE AREA_NAME = 'AFLPAL';"""
        ).count()
        > 0,
    )
    print(
        "AFL_PACKAGES",
        connection_context.sql(
            """SELECT * FROM "SYS"."AFL_PACKAGES" WHERE AREA_NAME = 'AFLPAL';"""
        ).count()
        > 0,
    )
    print(
        "AFL_FUNCTIONS",
        connection_context.sql(
            """SELECT * FROM "SYS"."AFL_FUNCTIONS" WHERE AREA_NAME = 'AFLPAL';"""
        ).count()
        > 0,
    )
    print("Checking if user exists...")
    users = connection_context.sql(f'SELECT USER_NAME FROM "SYS"."USERS"').collect()
    if username not in list(users["USER_NAME"]):
        print(
            f"User {username} does not exist, creating new user with default password "
            f'"newUser1533HelloWorld"...'
        )
        cursor.execute(f"CREATE USER {username} PASSWORD newUser1533HelloWorld;")
        print("Done")
        cursor.execute(f"ALTER USER {username} DISABLE PASSWORD LIFETIME;")
        print(f"""Changing {username}'s password to yours...""")
        cursor.execute(f"ALTER USER {username} PASSWORD {password};")
        print("Done")
    else:
        print(f"User {username} exists")
    print(f"Granting roles to {username} to execute PAL functions...")
    cursor.execute(f"""GRANT "AFL__SYS_AFL_AFLPAL_EXECUTE" TO {username}""")
    cursor.execute(
        f"""GRANT "AFL__SYS_AFL_AFLPAL_EXECUTE_WITH_GRANT_OPTION" TO {username}"""
    )
    print("Granting other roles...")
    cursor.execute(f"GRANT USER ADMIN, CATALOG READ TO {username}")
    print(f"Done.")
