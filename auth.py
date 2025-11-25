import sqlite3
import bcrypt

DB_PATH = "users.db"

def init_user_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def signup_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    try:
        c.execute("INSERT INTO users VALUES (?,?)", (email, hashed_pw))
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Email already registered!"
    finally:
        conn.close()

def login_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()

    if row and bcrypt.checkpw(password.encode(), row[0]):
        return True, "Login successful!"
    return False, "Invalid email / password!"
