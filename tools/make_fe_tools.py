from langchain.tools import tool

def make_fe_tools(state):

    @tool
    def create_ratio_feature(input: dict) -> dict:
        """Create a new feature as a ratio of two numeric columns.

        Args:
            input: Dictionary containing:
                - numerator: name of numerator column
                - denominator: name of denominator column
                - new_column: name of new feature
        """
        num = input["numerator"]
        den = input["denominator"]
        new_col = input["new_column"]

        df = state.df

        if num not in df.columns or den not in df.columns:
            return {"status": "error", "message": "Column not found"}

        df[new_col] = df[num] / (df[den] + 1e-9)

        state.df = df
        state.features.append(new_col)

        return {"status": "ok", "feature": new_col}
