sql = '''SELECT
    a.host,
    a.psm,
    COALESCE(NULLIF(a.normalized_path, ''), NULLIF(a.original_path, '')) as api_path,
    a.http_method,
    COUNT(DISTINCT a.employee_username) as emp_count,
    COUNT(DISTINCT a.waf_id) as traffic_count,
    COUNT(DISTINCT bytedatatag_id) as tag_count,
    concat_ws(',', collect_set(tag_name)) AS tag_names
FROM crystal_aplus.aplus_row_og_record_http_request_by_direction AS a
JOIN crystal_insight.dwd_row_og_http_policy_record_joined_hi AS b ON a.original_path = b.original_path AND a.http_method = b.http_method
WHERE a.date = "20250606"
    AND b.date = "20250606"
    AND a.employee_username IS NOT NULL
GROUP BY a.host, a.psm, 
    COALESCE(NULLIF(a.normalized_path, ''), NULLIF(a.original_path, '')),
    a.http_method
ORDER BY emp_count DESC, traffic_count DESC
LIMIT 100;'''


sql2 = '''
SELECT
    host,
    psm,
    COALESCE(NULLIF(normalized_path, ''), NULLIF(original_path, '')) AS api_path,
    http_method,
    COUNT(DISTINCT employee_username) as emp_count,
    COUNT(DISTINCT waf_id) as traffic_count,
    COUNT(DISTINCT bytedatatag_id) as tag_count,
    concat_ws(',', collect_set(tag_name)) AS tag_names
FROM crystal_insight.dwd_row_og_http_policy_record_joined_hi
WHERE date >= "20250606"
    AND date <= "20250606"
    AND employee_username IS NOT NULL
GROUP BY host, psm, 
    COALESCE(NULLIF(normalized_path, ''), NULLIF(original_path, '')),
    http_method
SORT BY tag_count DESC, emp_count DESC, traffic_count DESC
LIMIT 2147483647;
'''

sql3 = '''
WITH a_filtered AS (
  SELECT * FROM crystal_aplus.aplus_row_og_record_http_request_by_direction
  WHERE date = '20250606' AND employee_username IS NOT NULL
),
b_filtered AS (
  SELECT * FROM crystal_insight.dwd_row_og_http_policy_record_joined_hi
  WHERE date = '20250606'
)
SELECT
  a.host,
  a.psm,
  COALESCE(NULLIF(a.normalized_path, ''), NULLIF(a.original_path, '')) AS api_path,
  a.http_method,
  COUNT(DISTINCT a.employee_username) AS emp_count,
  COUNT(DISTINCT a.waf_id) AS traffic_count,
  COUNT(DISTINCT bytedatatag_id) AS tag_count,
  concat_ws(',', collect_set(tag_name)) AS tag_names
FROM a_filtered AS a
JOIN b_filtered AS b
  ON a.original_path = b.original_path AND a.http_method = b.http_method
GROUP BY a.host, a.psm, api_path, a.http_method
ORDER BY emp_count DESC, traffic_count DESC
LIMIT 100;
'''


sqls = [sql, sql2,sql3]
for sql in sqls:
    # 去除换行并压成一行
    flattened_sql = ' '.join(sql.replace("\"","'").split())
    print(flattened_sql)




ss = "SELECT a.host, a.psm, COALESCE(NULLIF(a.normalized_path, ''), NULLIF(a.original_path, '')) as api_path, a.http_method, COUNT(DISTINCT a.employee_username) as emp_count, COUNT(DISTINCT a.waf_id) as traffic_count, COUNT(DISTINCT bytedatatag_id) as tag_count, concat_ws(',', collect_set(tag_name)) AS tag_names FROM crystal_aplus.aplus_row_og_record_http_request_by_direction AS a JOIN crystal_insight.dwd_row_og_http_policy_record_joined_hi AS b ON a.original_path = b.original_path AND a.http_method = b.http_method WHERE a.date = '20250606' AND b.date = '20250606' AND a.employee_username IS NOT NULL GROUP BY a.host, a.psm, COALESCE(NULLIF(a.normalized_path, ''), NULLIF(a.original_path, '')), a.http_method ORDER BY emp_count DESC, traffic_count DESC LIMIT 100"
