# File: src/Test_red/app_backend/sql_utils.py

import json
from .api_clients import call_together_ai


def generate_sql_query(question: str, df, column_analysis: dict) -> str:
    """Generate SQL query for specific analysis"""
    
    # Build table schema
    table_schema = []
    for col, info in column_analysis.items():
        col_type = "TEXT"
        if info['likely_purpose'] in ['financial', 'performance_metric', 'volume_metric', 'numeric']:
            col_type = "DOUBLE"
        elif info['likely_purpose'] == 'temporal':
            col_type = "DATE"
        table_schema.append(f"`{col}` {col_type}")

    # Get sample data for better context
    sample_data = {}
    for col in list(column_analysis.keys())[:5]:  # Limit to 5 columns
        try:
            sample_values = df[col].dropna().head(3).tolist()
            sample_data[col] = sample_values
        except:
            sample_data[col] = []

    prompt = f"""
    Generate a SQL query to answer: "{question}"
    
    Table: marketing_data
    Schema: {', '.join(table_schema)}
    
    Column purposes:
    {json.dumps({col: info['likely_purpose'] for col, info in column_analysis.items()}, indent=2)}
    
    Sample data:
    {json.dumps(sample_data, indent=2, default=str)}
    
    RULES:
    - Return ONLY the SQL query, no explanations
    - Use appropriate aggregations and filters
    - Handle NULL values with COALESCE if needed
    - Use backticks around column names
    - Limit results to 100 rows if no specific limit mentioned
    - Use proper date functions if dealing with dates
    
    SQL Query:
    """
    
    try:
        sql = call_together_ai(prompt, max_tokens=400, temperature=0.0)
        
        # Clean up the SQL response
        sql = sql.strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ["```sql", "```", "SQL:", "Query:", "sql"]
        for prefix in prefixes_to_remove:
            if sql.lower().startswith(prefix.lower()):
                sql = sql[len(prefix):].strip()
        
        # Remove trailing semicolon and clean up
        sql = sql.rstrip(';').strip()
        
        # Basic validation - check if it looks like SQL
        if not any(keyword in sql.lower() for keyword in ['select', 'from', 'where', 'group', 'order']):
            return "Error: Generated query doesn't appear to be valid SQL"
        
        return sql
        
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"


def validate_sql_query(sql: str) -> bool:
    """Basic SQL query validation"""
    if not sql or not sql.strip():
        return False
    
    sql_lower = sql.lower().strip()
    
    # Must contain SELECT and FROM
    if 'select' not in sql_lower or 'from' not in sql_lower:
        return False
    
    # Should reference our table
    if 'marketing_data' not in sql_lower:
        return False
    
    # Basic security check - no dangerous keywords
    dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
    if any(keyword in sql_lower for keyword in dangerous_keywords):
        return False
    
    return True


def format_sql_result(result_df, question: str) -> str:
    """Format SQL query results for display"""
    if result_df.empty:
        return "No data found matching your query."
    
    # Get basic info about results
    total_rows = len(result_df)
    total_cols = len(result_df.columns)
    
    summary = f"**Query Results:** {total_rows:,} rows × {total_cols} columns\n\n"
    
    # Add basic insights if numeric data
    numeric_cols = result_df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary += "**Key Numbers:**\n"
        for col in numeric_cols[:3]:  # Limit to 3 columns
            try:
                total = result_df[col].sum()
                avg = result_df[col].mean()
                summary += f"• {col}: Total = {total:,.2f}, Average = {avg:,.2f}\n"
            except:
                continue
    
    return summary