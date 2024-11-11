import sqlite3

def fetch_responses():
    # Connect to the database
    conn = sqlite3.connect('user_responses.db')
    c = conn.cursor()

    # Query the data from the 'responses' table
    c.execute("SELECT * FROM responses")
    
    # Fetch all the rows
    rows = c.fetchall()

    # Close the connection
    conn.close()

    # Print the rows
    for row in rows:
        print(row)

# Call the function to fetch and display data
fetch_responses()
