import psycopg2
from psycopg2 import Error

def get_db_connection():
    try:
        connection = psycopg2.connect(
            database="AR_Assembly_DB",
            user="postgres",
            password="postgres",
            host="127.0.0.1",
            port="5432"
        )
        return connection
    except Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def get_equipment_details(serial_number):
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM equipment WHERE serial_number = %s", (serial_number,))
            equipment = cursor.fetchone()
            cursor.close()
            connection.close()
            if equipment:
                return {
                    "serial_number": equipment[0],
                    "equipment_name": equipment[1],
                    "asset_id": equipment[2],
                    "part_number": equipment[3],
                    "location_name": equipment[4],
                    "documentation": equipment[5]
                }
            else:
                return None  # Aucun équipement trouvé
        except Error as e:
            print(f"Erreur lors de la requête : {e}")
            return None
    return None

# Example: Create table equipments if not exists
def init_db():
    connection = get_db_connection()
    if connection:
        cursor = connection.cursor()
        cursor.execute("""
        """)
        connection.commit()
        cursor.close()
        connection.close()

if __name__ == "__main__":
    init_db()