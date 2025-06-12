-- Database: AR_Assembly_DB

-- DROP DATABASE IF EXISTS "AR_Assembly_DB";

CREATE DATABASE "AR_Assembly_DB"
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    LOCALE_PROVIDER = 'libc'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;

\c AR_Assembly_DB

CREATE TABLE equipment (
    serial_number VARCHAR(20) PRIMARY KEY,
    equipment_name VARCHAR(50),
    asset_id VARCHAR(20),
    part_number VARCHAR(20),
    location_name VARCHAR(50),
    documentation TEXT
);

INSERT INTO equipment (serial_number, equipment_name, asset_id, part_number, location_name, documentation)
VALUES ('SN-IM2025001', 'Industrial Motor', 'IM-2025-001', '45678-IM', 'Production Line B', 'Guide pour Industrial Motor : Instructions de maintenance et dépannage.');