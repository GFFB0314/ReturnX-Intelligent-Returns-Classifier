-- Create Schema
CREATE SCHEMA IF NOT EXISTS retail_returns;

-- Drop table if exists
DROP TABLE IF EXISTS retail_returns.raw_reviews;

-- Create Table
CREATE TABLE retail_returns.raw_reviews (
    index_id INTEGER,
    clothing_id INTEGER,
    age INTEGER,
    title TEXT,
    review_text TEXT,
    rating INTEGER,
    recommended_ind INTEGER,
    positive_feedback_count INTEGER,
    division_name TEXT,
    department_name TEXT,
    class_name TEXT
);
