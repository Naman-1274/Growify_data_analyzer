import json
from .api_clients import call_together_ai

def generate_sql_query(question: str, df, column_analysis: dict) -> str:
    """Generate SQL query for specific analysis"""
    table_schema = []
    for col, info in column_analysis.items():
        col_type = "TEXT"
        if info['likely_purpose'] in ['financial', 'performance_metric', 'volume_metric', 'numeric']:
            col_type = "DOUBLE"
        elif info['likely_purpose'] == 'temporal':
            col_type = "DATE"
        table_schema.append(f"{col} {col_type}")

    prompt = f"""
    Generate a SQL query to answer: \"{question}\"
    Table: marketing_data
    Schema: {', '.join(table_schema)}  
    Column purposes:
    {json.dumps({col: info['likely_purpose'] for col, info in column_analysis.items()}, indent=2)}
    Return only the SQL query, no explanations.
    Use appropriate aggregations and filters.
    """
    sql = call_together_ai(prompt, max_tokens=300)
    return sql.strip().rstrip(';')