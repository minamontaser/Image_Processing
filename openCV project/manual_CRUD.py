import sqlite3 as sql
import os
import datetime

con = sql.connect('students.db')
cur =con.cursor()

#Reading data
def get_class_id(class_name: str):
    cur.execute("SELECT ClassID FROM Classes WHERE ClassName = ?", (class_name,))
    result = cur.fetchone()

    return result[0] if result else None

def get_student_id(name: str):
    cur.execute("SELECT StudentID FROM Students WHERE Name = ?", (name,))
    result = cur.fetchone()

    return result[0] if result else None

def get_session_id(class_id: int, date: str):
    cur.execute("SELECT SessionID FROM Sessions WHERE ClassID = ? AND Date = ?", (class_id, date))
    result = cur.fetchone()

    return result[0] if result else None

def get_attendance(student_id: int, session_id: int):
    cur.execute("SELECT Status FROM Attendance WHERE StudentID = ? AND SessionID = ?", (student_id, session_id))
    result = cur.fetchone()

    return result[0] if result else None

def get_attendance_report(session_id: int):
    cur.execute("""
    SELECT s.Name, a.Status
    FROM Attendance a
    JOIN Students s ON a.StudentID = s.StudentID
    WHERE a.SessionID = ?
    """, (session_id,))

    return cur.fetchall()

def get_performance_report(session_id: int):
    cur.execute("""
    SELECT s.Name, p.FocusScore, p.AttentionTime, p.DistractedTime, p.TotalVisibleTime
    FROM Performance p
    JOIN Students s ON p.StudentID = s.StudentID
    WHERE p.SessionID = ?
    """, (session_id,))

    return cur.fetchall()

#Inserting data
def insert_class(class_name: str, schedule_start: str, schedule_end: str):
    cur.execute("""
    INSERT INTO Classes (ClassName, ScheduleStart, ScheduleEnd)
    VALUES (?, ?, ?)
    """, (class_name, schedule_start, schedule_end))

    con.commit()

def insert_student(name: str, email: str, phone: str, parent_phone: str, level: str, class_id: int):
    cur.execute("""
    INSERT INTO Students (Name, Email, Phone, ParentPhone, Level, ClassID)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (name, email, phone, parent_phone, level, class_id))

    con.commit()

def insert_session(class_id: int, date: str, start_time: str, end_time: str):
    cur.execute("""
    INSERT INTO Sessions (ClassID, Date, StartTime, EndTime)
    VALUES (?, ?, ?, ?)
    """, (class_id, date, start_time, end_time))

    con.commit()

def insert_attendance(session_id: int, student_id: int, status: str, first_seen: str, last_seen: str):
    cur.execute("""
    INSERT INTO Attendance (SessionID, StudentID, Status, FirstSeen, LastSeen)
    VALUES (?, ?, ?, ?, ?)
    """, (session_id, student_id, status, first_seen, last_seen))

    con.commit()

#Updating data
def update_class(class_id: int, class_name: str, schedule_start: str, schedule_end: str):
    cur.execute("""
    UPDATE Classes
    SET ClassName = ?,
    ScheduleStart = ?,
    ScheduleEnd = ?
    WHERE ClassID = ?
    """, (class_name, schedule_start, schedule_end, class_id))

    con.commit()

def update_student(student_id: int, name: str, email: str, phone: str, parent_phone: str, level: str, class_id: int):
    cur.execute("""
    UPDATE Students
    SET Name = ?,
    Email = ?,
    Phone = ?,
    ParentPhone = ?,
    Level = ?,
    ClassID = ?
    WHERE StudentID = ?
    """, (name, email, phone, parent_phone, level, class_id, student_id))

    con.commit()

def update_session(session_id: int, class_id: int, date: str, start_time: str, end_time: str):
    cur.execute("""
    UPDATE Sessions
    SET ClassID = ?,
    Date = ?,
    StartTime = ?,
    EndTime = ?
    WHERE SessionID = ?
    """, (class_id, date, start_time, end_time, session_id))

    con.commit()