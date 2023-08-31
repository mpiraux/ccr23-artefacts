WITH MatchedDownloads as (
  WITH HTTPDownloads as (
    SELECT 
      A.af,
      A.prb_id,
      B.prb_id as anchor_id,
      A.start_time,
      A.http_duration,
    FROM `ripencc-atlas.measurements.http` AS A
    JOIN `ripencc-atlas.probes.probes` AS B ON B.is_anchor AND ((A.af = 4 AND A.dst_addr = B.addr_v4) OR (A.af = 6 AND A.dst_addr = B.addr_v6))
    WHERE http_status = 200 AND
    "2023-06-01" <= A.start_time AND A.start_time < "2023-06-08"
  )

  SELECT
    A.prb_id,
    A.anchor_id,
    A.start_time,
    A.http_duration as v4_duration,
    B.http_duration as v6_duration,
    A.http_duration / B.http_duration as v4_v6_ratio,
  FROM HTTPDownloads AS A
  JOIN HTTPDownloads AS B ON 
    A.prb_id = B.prb_id 
    AND A.anchor_id = B.anchor_id 
    AND A.af = 4 
    AND A.af != B.af
    AND ABS(TIMESTAMP_DIFF(A.start_time, B.start_time, MINUTE)) < 2
  ORDER BY 1, 3
)

SELECT
  *
FROM MatchedDownloads
