def generate_processing_schema(minmax_calls, maxpoly_calls, mxp_calls, mnp_calls, s_cond_parts, ct_inits, tones_parts):
    tones_def = (
        f"t_{{ones}}=\\left\\{{{','.join(tones_parts)}\\right\\}}"
        if len(tones_parts) > 1
        else f"t_{{ones}}={tones_parts[0]}"
    )

    lines = [
        f"m_{{inmax}}=\\left[{','.join(minmax_calls)}\\right]",
        f"m_{{axpoly}}=6\\max\\left({','.join(maxpoly_calls)}\\right)",
        f"M=\\left\\{{m_{{inmax}}.x\\le t_{0}<m_{{inmax}}.y,0\\right\\}}",
        f"m_{{axpitch}}=\\max\\left({','.join(mxp_calls)}\\right)",
        f"m_{{inpitch}}=\\min\\left({','.join(mnp_calls)}\\right)",
        f"s_{{upercond}}={','.join(s_cond_parts)}",
        *ct_inits,
        tones_def,
        "d_{uration}=\\max\\left(m_{{inmax}}.y\\right)",
    ]
    return "\n".join(lines)
