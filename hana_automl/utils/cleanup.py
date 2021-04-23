from hdbcli import dbapi
from credentials import port, host, user, password

"""For developers only. Cleanup database by deleting all test tables. Be careful!"""

conn = dbapi.connect(address=host, user=user, password=password, port=port)
cursor = conn.cursor()
for i in open("utils/tables.txt").readlines():
    i = i[:-1]  # remove '\n'
    print("Removing", i)
    cursor.execute(f'DROP TABLE"{i}";')
conn.close()
open("utils/tables.txt", "w").close()  # erase file contents
