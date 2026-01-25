-- 1. Create a clean view that removes the NULL texts
CREATE OR REPLACE VIEW retail_returns.clean_reviews_v AS 
SELECT * 
FROM retail_returns.raw_reviews
WHERE review_text IS NOT NULL 
  AND review_text != '';

-- 2. THE LOGIC ENGINE
-- We will search for keywords to categorize the returns.
-- This simulates a "Business Rule" layer.

CREATE OR REPLACE VIEW retail_returns.labeled_reviews_v AS
SELECT 
    index_id,
    clothing_id,
    age,
    review_text,
    rating,
    
    -- LOGIC START --
    CASE -- Return Reason Classification
        -- CATEGORY 1: PRODUCT DEFECT (The most expensive return)
        -- Keywords: rip, tear, hole, broke, zipper, stain, poor quality
        WHEN LOWER(review_text) LIKE '%rip%' 
          OR LOWER(review_text) LIKE '%tear%' 
          OR LOWER(review_text) LIKE '%hole%' 
          OR LOWER(review_text) LIKE '%zipper%' 
          OR LOWER(review_text) LIKE '%stain%' 
          OR LOWER(review_text) LIKE '%cheap%'
          OR LOWER(review_text) LIKE '%poor quality%'
          THEN 'Defect'

        -- CATEGORY 2: SIZING ISSUE (The most common return)
        -- Keywords: small, large, tight, loose, fit, size
        WHEN LOWER(review_text) LIKE '%small%' 
          OR LOWER(review_text) LIKE '%large%' 
          OR LOWER(review_text) LIKE '%tight%' 
          OR LOWER(review_text) LIKE '%loose%' 
          OR LOWER(review_text) LIKE '%fit%' 
          THEN 'Sizing'
          
        -- CATEGORY 3: MISMATCH / DISLIKE / CHANGE OF MIND (General churning)
        -- Keywords: color, look, picture, fabric
        WHEN LOWER(review_text) LIKE '%color%' 
          OR LOWER(review_text) LIKE '%fabric%' 
          OR LOWER(review_text) LIKE '%material%' 
          OR LOWER(review_text) LIKE '%look%' 
          THEN 'Style'

        -- EVERYTHING ELSE
        ELSE 'Other'
    END AS return_category
    -- LOGIC END --

FROM retail_returns.clean_reviews_v;