import sqlite3 as sql

def dataBase(db_path: str = 'students.db'):
    con = sql.connect(db_path)
    cur = con.cursor()

    # Core safety
    con.execute("PRAGMA foreign_keys = ON")

    # Performance (balanced for small DB)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute("PRAGMA temp_store = MEMORY")
    con.execute("PRAGMA cache_size = -2000")
    con.execute("PRAGMA mmap_size = 134217728")

    # Security
    con.execute("PRAGMA secure_delete = ON")
    con.execute("PRAGMA recursive_triggers = ON")

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS Classes (
        ClassID INTEGER PRIMARY KEY AUTOINCREMENT,
        ClassName TEXT,
        ScheduleStart TIME,
        ScheduleEnd TIME
    );

    CREATE TABLE IF NOT EXISTS Students (
        StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT NOT NULL,
        Email TEXT UNIQUE,
        Phone TEXT,
        ParentPhone TEXT,
        Level TEXT,
        ClassID INTEGER,
        FaceEncoding BLOB,
        CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ClassID) REFERENCES Classes(ClassID) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS Sessions (
        SessionID INTEGER PRIMARY KEY AUTOINCREMENT,
        ClassID INTEGER NOT NULL,
        Date DATE,
        StartTime DATETIME,
        EndTime DATETIME,
        FOREIGN KEY (ClassID) REFERENCES Classes(ClassID) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS Attendance (
        AttendanceID INTEGER PRIMARY KEY AUTOINCREMENT,
        SessionID INTEGER NOT NULL,
        StudentID INTEGER NOT NULL,
        Status TEXT,
        FirstSeen DATETIME,
        LastSeen DATETIME,
        FOREIGN KEY (SessionID) REFERENCES Sessions(SessionID) ON DELETE CASCADE,
        FOREIGN KEY (StudentID) REFERENCES Students(StudentID) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS Performance (
        PerformanceID INTEGER PRIMARY KEY AUTOINCREMENT,
        SessionID INTEGER NOT NULL,
        StudentID INTEGER NOT NULL,
        FocusScore REAL,
        AttentionTime INTEGER,
        DistractedTime INTEGER,
        TotalVisibleTime INTEGER,
        FOREIGN KEY (SessionID) REFERENCES Sessions(SessionID) ON DELETE CASCADE,
        FOREIGN KEY (StudentID) REFERENCES Students(StudentID) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS TrackingLogs (
        LogID INTEGER PRIMARY KEY AUTOINCREMENT,
        StudentID INTEGER,
        SessionID INTEGER,
        Timestamp DATETIME,
        State TEXT,
        FOREIGN KEY (StudentID) REFERENCES Students(StudentID) ON DELETE CASCADE,
        FOREIGN KEY (SessionID) REFERENCES Sessions(SessionID) ON DELETE CASCADE
    );
    """)

    con.commit()
    con.close()


if __name__ == "__main__":
    dataBase()