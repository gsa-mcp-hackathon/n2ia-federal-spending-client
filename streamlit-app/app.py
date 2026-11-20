
"""Federal Spending & Investment Intelligence — Databricks Streamlit client."""

import os
import sys
import json
from urllib.parse import urlparse

# Install existing diagnostics before loading application code.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import app_diagnostics  # noqa: E402

app_diagnostics.install_diagnostics()

import streamlit as st
from databricks.sdk import WorkspaceClient


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

#SUPERVISOR_ENDPOINT = "mas-c66da9c5-endpoint"
SUPERVISOR_ENDPOINT = "mas-7157a657-endpoint"

TRUSTED_MCP_TOOLS = {
    "contract_awards",
    "search_contract_awards",
    "retrieve_service_category",
    "analyze_contract_overlap",
    "compare_agency_it_spending",
}

TRUSTED_MCP_SERVER = "mcp-n2ia-server-app"
MAX_AUTO_APPROVAL_ROUNDS = 10

st.set_page_config(
    page_title="Federal Spending Intelligence",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    /* Reduce top whitespace without clipping the header */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1rem !important;
    }

    /* Keep title and icon fully visible */
    h1 {
        margin-top: 0 !important;
        padding-top: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Federal Spending & Investment Intelligence")

st.caption(
    "Explore federal IT contract spending, identify potential "
    "contract overlaps, and investigate acquisition patterns "
    "using USAspending.gov data and Databricks AI."
)


# ---------------------------------------------------------
# Databricks connection
# ---------------------------------------------------------

# Uses credentials automatically supplied to the App.
# No hard-coded access tokens are required.

@st.cache_resource
def get_client():
    workspace = WorkspaceClient()
    return workspace.serving_endpoints.get_open_ai_client()


client = get_client()


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Full Responses API input history, including tool approvals.
if "conversation_input" not in st.session_state:
    st.session_state.conversation_input = []

if "pending_response" not in st.session_state:
    st.session_state.pending_response = None

if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

if "overlap_analyses" not in st.session_state:
    st.session_state.overlap_analyses = {}
if "overlap_expected" not in st.session_state:
    st.session_state.overlap_expected = []
if "overlap_queue" not in st.session_state:
    st.session_state.overlap_queue = []
if "overlap_active" not in st.session_state:
    st.session_state.overlap_active = None


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_approval_requests(response):
    """Find MCP approval requests in a Supervisor response."""
    return [
        item
        for item in response.output
        if item.type == "mcp_approval_request"
    ]


def response_items(response):
    """Convert response output into serializable input items."""
    return [
        item.model_dump(exclude_none=True)
        for item in response.output
    ]

def extract_overlap_analysis(items):
    """Find structured overlap data in MCP output; never parse narrative prose."""
    def visit(value, depth=0):
        if depth > 8:
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return None
        if isinstance(value, list):
            for item in reversed(value):
                found = visit(item, depth + 1)
                if found is not None:
                    return found
        if isinstance(value, dict):
            # Preserve the outer query/coverage metadata needed for FY labels.
            inner = value.get("analysis")
            if (isinstance(inner, dict) and isinstance(value.get("query"), dict)
                    and isinstance(inner.get("groups"), list)
                    and isinstance(inner.get("ranked_candidates"), list)):
                return {**inner, "query": value["query"],
                        "coverage": value.get("coverage"),
                        "complete": value.get("complete"),
                        "retrieved_count": value.get("retrieved_count")}
            if (isinstance(value.get("groups"), list)
                    and isinstance(value.get("ranked_candidates"), list)):
                return value
            for key in ("analysis", "result", "output", "content", "text", "data", "structuredContent"):
                if key in value:
                    found = visit(value[key], depth + 1)
                    if found is not None:
                        return found
        return None

    for item in reversed(items):
        if item.get("type") not in ("mcp_call", "function_call_output") or item.get("name") != "analyze_contract_overlap":
            continue
        found = visit(item)
        if found is not None:
            return found
    return None


def analysis_label(analysis):
    """Identify a structured tool result by its actual query dates and category."""
    query = analysis.get("query") or {}
    start, end = str(query.get("start_date") or ""), str(query.get("end_date") or "")
    if len(start) >= 4 and len(end) >= 4 and start[5:10] == "10-01" and end[5:10] == "09-30":
        fy = f"FY{end[:4]}"
    else:
        fy = f"{start or 'Unknown start'} to {end or 'Unknown end'}"
    category = str(query.get("keyword") or "Category unspecified").strip()
    return f"{fy} · {category.title()}"


def collect_overlap_analyses(items):
    """Collect every successful tool output, not only the last fiscal year."""
    results = {}
    for item in items:
        if (item.get("type") not in ("mcp_call", "function_call_output")
                or item.get("name") != "analyze_contract_overlap"):
            continue
        analysis = extract_overlap_analysis([item])
        if analysis is not None:
            results[analysis_label(analysis)] = analysis
    return results


def contract_key(contract):
    """Stable identity for an award across repeated pairwise results."""
    return str(contract.get("source_award_key") or contract.get("award_id") or "").strip()


def build_preliminary_groups(candidates):
    """Connected components of candidate relationships; no claim of all-to-all similarity."""
    parent = {}
    contracts = {}
    edges = []

    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for pair in candidates:
        left, right = pair.get("left") or {}, pair.get("right") or {}
        a, b = contract_key(left), contract_key(right)
        if not a or not b or a == b:
            continue
        contracts.setdefault(a, left)
        contracts.setdefault(b, right)
        parent[root(b)] = root(a)
        edges.append((a, b, pair))

    components = {}
    for key in contracts:
        components.setdefault(root(key), {"contracts": [], "relationships": []})["contracts"].append(contracts[key])
    for a, b, pair in edges:
        components[root(a)]["relationships"].append(pair)
    return sorted(components.values(), key=lambda g: (-len(g["contracts"]), -len(g["relationships"])))


def show_contract_table(contracts):
    rows = []
    for contract in contracts:
        rows.append({
            "Agency": contract.get("awarding_agency") or "—",
            "Award ID": contract.get("award_id") or "—",
            "Contractor": contract.get("contractor") or "—",
            "Start": contract.get("performance_start_date") or "—",
            "End": contract.get("performance_end_date") or "—",
            "PSC": contract.get("psc") or "—",
            "NAICS": contract.get("naics") or "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def show_contract_details(contracts, key_prefix):
    """Show details inside a set without nesting Streamlit expanders."""
    st.markdown("**Contract details**")
    for i, contract in enumerate(contracts):
        title = f"{contract.get('award_id') or 'Unknown award'} · {contract.get('awarding_agency') or 'Unknown agency'}"
        st.markdown(f"**{title}**")
        st.write(contract.get("description") or "No description available")
        st.caption(f"Contractor: {contract.get('contractor') or 'Unknown'} | "
                   f"Performance: {contract.get('performance_start_date') or '?'} "
                   f"to {contract.get('performance_end_date') or '?'}")
        url = contract.get("source_url")
        if isinstance(url, str):
            parsed = urlparse(url)
            if parsed.scheme == "https" and (parsed.hostname == "usaspending.gov" or
                                               (parsed.hostname or "").endswith(".usaspending.gov")):
                st.link_button("View award on USAspending", url)
        st.divider()


def show_relationship_graph(contracts, relationships, key_prefix, qualified=False):
    """Visualize only evidenced edges; an indirect connection is not a direct match."""
    if len(contracts) > 25:
        st.caption("Network view omitted for groups above 25 contracts; relationship table remains available.")
        return
    keys = [contract_key(c) for c in contracts]
    if not all(keys) or len(set(keys)) != len(keys):
        st.caption("Network view unavailable because contract identifiers are incomplete.")
        return
    import json as _json
    nodes = {key: f"n{i}" for i, key in enumerate(keys)}
    dot = ['graph Overlap {', 'rankdir=LR;', 'node [shape=box, style="rounded,filled", fillcolor="#E9F1FA", color="#7D9ABB"];']
    for key, contract in zip(keys, contracts):
        label = f"{contract.get('awarding_agency') or 'Agency'}\n{contract.get('award_id') or key}"
        dot.append(f'{nodes[key]} [label={_json.dumps(label)}];')
    for rel in relationships:
        if qualified:
            a = str(rel.get("award_key_a") or rel.get("award_id_a") or "")
            b = str(rel.get("award_key_b") or rel.get("award_id_b") or "")
        else:
            a = contract_key(rel.get("left") or {})
            b = contract_key(rel.get("right") or {})
        if a in nodes and b in nodes and a != b:
            dot.append(f'{nodes[a]} -- {nodes[b]};')
    dot.append('}')
    try:
        st.graphviz_chart("\n".join(dot), use_container_width=True)
    except ImportError:
        st.caption("Network diagram requires the optional graphviz Python package; relationship evidence is shown below.")
    st.caption("Lines indicate reported direct relationships; contracts connected through another contract may not match each other.")


def render_overlap_dashboard(analysis):
    """Group-centric view of qualified and preliminary results; no inferred duplication."""
    st.divider()
    st.subheader("Potentially Related Contracts")
    groups = analysis.get("groups") or []
    ranked = analysis.get("ranked_candidates") or []
    qualified_edges = {
        frozenset((str(edge.get("award_key_a") or edge.get("award_id_a") or ""),
                   str(edge.get("award_key_b") or edge.get("award_id_b") or "")))
        for group in groups for edge in (group.get("relationships") or [])
    }
    candidates = [pair for pair in ranked
                  if frozenset((contract_key(pair.get("left") or {}),
                                contract_key(pair.get("right") or {}))) not in qualified_edges]
    preliminary_groups = build_preliminary_groups(candidates)

    a, b, c, d = st.columns(4)
    a.metric("Contracts analyzed", analysis.get("analyzed_awards", "—"))
    b.metric("Higher-confidence sets", analysis.get("group_count", len(groups)))
    c.metric("Related contract sets", len(preliminary_groups))
    d.metric("Potential relationships", analysis.get("ranked_candidate_count", len(ranked)))
    st.caption("Similarity signals are screening evidence, not proof of duplication, waste, or savings. "
               "Related sets connect contracts with reported similarities; not every contract in a set necessarily matches every other.")

    st.markdown("#### Higher-confidence related contracts")
    if not groups:
        st.info("No contract sets met the stricter similarity and service-function criteria.")
    for gi, group in enumerate(groups):
        contracts = group.get("contracts") or []
        relationships = group.get("relationships") or []
        functions = ", ".join(str(f).replace("_", " ").title()
                              for f in (group.get("service_functions") or [])) or "Related services"
        title = f"{group.get('group_id') or f'Group {gi + 1}'} · {functions} · {len(contracts)} contracts"
        with st.expander(title):
            st.caption(f"{len({c.get('awarding_agency') for c in contracts})} agencies · "
                       f"{len(relationships)} reported relationships")
            show_contract_table(contracts)
            if relationships:
                show_relationship_graph(contracts, relationships, f"qualified_{gi}", qualified=True)
                st.markdown("**Relationship evidence**")
                st.dataframe([{
                    "Contract A": edge.get("award_id_a"),
                    "Contract B": edge.get("award_id_b"),
                    "Shared functions": ", ".join(edge.get("shared_service_functions") or []),
                    "Semantic similarity": edge.get("semantic_similarity"),
                    "Overlap days": edge.get("overlap_days"),
                } for edge in relationships], use_container_width=True, hide_index=True)
            show_contract_details(contracts, f"qualified_{gi}")

    st.markdown("#### Other contracts worth reviewing")
    st.caption("These sets are based on preliminary matches. A set of three contracts may contain only two direct relationships.")
    if not preliminary_groups:
        st.info("No preliminary related contracts were found for this analysis.")
    for gi, group in enumerate(preliminary_groups):
        contracts = group["contracts"]
        relationships = group["relationships"]
        agency_count = len({c.get("awarding_agency") for c in contracts})
        title = (f"Related set {gi + 1} · {len(contracts)} contracts · "
                 f"{agency_count} agencies · {len(relationships)} relationships")
        with st.expander(title, expanded=(gi == 0)):
            show_contract_table(contracts)
            show_relationship_graph(contracts, relationships, f"preliminary_{gi}")
            st.markdown("**Why these contracts were linked**")
            st.dataframe([{
                "Contract A": (pair.get("left") or {}).get("award_id"),
                "Contract B": (pair.get("right") or {}).get("award_id"),
                "Lexical similarity": pair.get("lexical_similarity"),
                "Semantic similarity": pair.get("semantic_similarity"),
                "Overlap days": pair.get("overlap_days"),
                "Shared terms": ", ".join(pair.get("shared_terms") or []),
            } for pair in relationships], use_container_width=True, hide_index=True)
            show_contract_details(contracts, f"preliminary_{gi}")

    if analysis.get("ranked_candidates_truncated"):
        st.warning("Candidate results are truncated; preliminary clusters may be incomplete.")
    if analysis.get("groups_truncated"):
        st.warning("Qualified group results are truncated.")

def submit_to_supervisor(input_items):
    """Send a request to the Databricks Supervisor Agent."""

    return client.responses.create(
        model=SUPERVISOR_ENDPOINT,
        input=input_items,
    )

def handle_supervisor_response(response, submitted_input):
    """Handle Supervisor responses and auto-approve trusted MCP tools."""

    current_response = response
    current_input = submitted_input

    for _ in range(MAX_AUTO_APPROVAL_ROUNDS):
        approvals = get_approval_requests(current_response)

        items = response_items(current_response)
        st.session_state.overlap_analyses.update(collect_overlap_analyses(items))
        updated_input = current_input + items

        if not approvals:
            answer = current_response.output_text or (
                "The Supervisor completed the request but did not "
                "return a text response."
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "is_overlap_report": bool(st.session_state.overlap_expected),
            })

            st.session_state.conversation_input = updated_input
            st.session_state.overlap_active = None
            st.session_state.pending_response = None
            st.session_state.pending_input = None
            return

        # Only auto-approve when EVERY pending request is trusted.
        trusted = all(
            getattr(request, "server_label", None)
            == TRUSTED_MCP_SERVER
            and getattr(request, "name", None)
            in TRUSTED_MCP_TOOLS
            for request in approvals
        )

        if not trusted:
            # Fall back to existing manual approval UI.
            st.session_state.pending_response = current_response
            st.session_state.pending_input = updated_input
            return

        # Approve trusted requests.
        for request in approvals:
            updated_input.append({
                "type": "mcp_approval_response",
                "approval_request_id": request.id,
                "approve": True,
            })

        current_input = updated_input
        current_response = submit_to_supervisor(current_input)

    # Safety limit: do not continue automatically indefinitely.
    st.session_state.pending_response = current_response
    st.session_state.pending_input = (
        current_input + response_items(current_response)
    )

def resolve_approval(approve):
    """Approve or deny outstanding MCP requests."""

    response = st.session_state.pending_response
    continued_input = list(st.session_state.pending_input)

    approvals = get_approval_requests(response)

    if not approvals:
        return

    for request in approvals:
        continued_input.append({
            "type": "mcp_approval_response",
            "approval_request_id": request.id,
            "approve": approve,
        })

    # The Supervisor may return another approval request.
    # handle_supervisor_response supports that case.
    next_response = submit_to_supervisor(continued_input)

    handle_supervisor_response(
        next_response,
        continued_input,
    )


# ---------------------------------------------------------
# Investigation filters and tabs
# ---------------------------------------------------------

# Set this once in the client App environment. Subsequent JSON edits need no redeploy.
UI_CONFIG_VOLUME_PATH = os.environ.get(
    "UI_CONFIG_VOLUME_PATH",
    "/Volumes/users/satish_venkataraman/federal_spending_ui/ui_config.json",
)

FALLBACK_UI_CONFIG = {
    "agencies": ["GSA", "DHS", "Treasury"],
    "fiscal_years": ["FY2022", "FY2023", "FY2024", "FY2025", "FY2026"],
    "it_categories": ["Cybersecurity", "Cloud computing", "IT support"],
    "defaults": {
        "agencies": ["GSA", "DHS", "Treasury"],
        "fiscal_years": ["FY2024", "FY2025", "FY2026"],
        "it_categories": ["Cybersecurity"],
    },
}


def validate_ui_config(data):
    """Validate external config; preserve option order and reject invalid shapes."""
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    result = {}
    for key in ("agencies", "fiscal_years", "it_categories"):
        options = data.get(key)
        if (not isinstance(options, list) or not options or
                any(not isinstance(x, str) or not x.strip() for x in options) or
                len(options) != len(set(options))):
            raise ValueError(f"Invalid or duplicate options in {key}")
        result[key] = options
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a JSON object")
    result["defaults"] = {}
    for key in ("agencies", "fiscal_years", "it_categories"):
        selected = defaults.get(key, [])
        if not isinstance(selected, list) or any(x not in result[key] for x in selected):
            raise ValueError(f"Invalid defaults for {key}")
        result["defaults"][key] = selected
    return result


@st.cache_data(ttl=60, show_spinner=False)
def load_ui_config(volume_path):
    """Read a Unity Catalog Volume file via Databricks Files API."""
    workspace = WorkspaceClient()
    response = workspace.files.download(volume_path)
    try:
        raw = response.contents.read()
    finally:
        response.contents.close()
    return validate_ui_config(json.loads(raw.decode("utf-8")))


try:
    ui_config = load_ui_config(UI_CONFIG_VOLUME_PATH)
except Exception as exc:
    ui_config = FALLBACK_UI_CONFIG
    st.warning(
        "Dropdown configuration could not be loaded from the Volume; "
        "using built-in options. Check the App resource permissions and JSON file. "
        f"Details: {type(exc).__name__}: {exc}"
    )

AGENCY_OPTIONS = ui_config["agencies"]
FY_OPTIONS = ui_config["fiscal_years"]
CATEGORY_OPTIONS = ui_config["it_categories"]

# Streamlit multiselect state must not contain values removed from the config.
for state_key, config_key in (
    ("selected_agencies", "agencies"),
    ("selected_years", "fiscal_years"),
    ("selected_categories", "it_categories"),
):
    allowed = ui_config[config_key]
    if state_key not in st.session_state:
        st.session_state[state_key] = ui_config["defaults"][config_key].copy()
    else:
        st.session_state[state_key] = [
            value for value in st.session_state[state_key] if value in allowed
        ]

st.subheader("Investigation filters")
col_agencies, col_years, col_categories = st.columns(3)
with col_agencies:
    agencies = st.multiselect(
        "Agencies", AGENCY_OPTIONS, key="selected_agencies",
        help="Current overlap MCP tools accept up to three agencies.",
    )
with col_years:
    fiscal_years = st.multiselect(
        "Fiscal years", FY_OPTIONS, key="selected_years",
        help="Nonconsecutive fiscal years are analyzed separately to avoid including unselected years.",
    )
with col_categories:
    categories = st.multiselect(
        "IT service categories", CATEGORY_OPTIONS, key="selected_categories",
        help="Each category is analyzed separately using the current backend tools.",
    )

valid_filters = bool(agencies and fiscal_years and categories)
if not valid_filters:
    st.warning("Select at least one agency, fiscal year, and IT service category.")


def build_analysis_question(kind, scope=None, extra=""):
    """Create explicit, bounded instructions without changing MCP tool contracts."""
    selected = ", ".join(agencies)
    fy_text = ", ".join(sorted(fiscal_years))
    category_text = ", ".join(categories)
    common = (
        f"Selected agencies: {selected}. Selected fiscal years: {fy_text}. "
        f"Selected IT categories: {category_text}. "
        "Only analyze the selected years and categories; do not silently include "
        "unselected years or categories. If the tools cannot honor these filters "
        "exactly, state the limitation rather than approximating without disclosure. "
        "Use USAspending.gov evidence, cite source award URLs when available, "
        "and disclose retrieval limits and incomplete coverage. "
    )
    if kind == "overlap":
        analysis = (
            "Investigate potential contract overlap. For each selected category "
            "and fiscal year, use the consolidated analyze_contract_overlap MCP tool. "
            "Use its lexical and semantic results, ranked candidates, and "
            "screened NetworkX contract groups when available. "
            "Identify potential related awards with overlapping performance "
            "periods, show evidence and explain why human review is needed. "
            "Do not assert confirmed duplication or savings. "
            + ("Compare awards across different selected agencies. " if scope == "cross" else
               "Include comparisons within and across selected agencies. ")
        )
    else:
        analysis = (
            "Compare agency contract spending and acquisition activity using "
            "available MCP tools. Report only metrics supported by retrieved "
            "records; distinguish award counts, obligations and expenditures. "
            "Do not infer efficiency or cost savings from spending alone. "
            "If complete comparable spending metrics are unavailable, say so "
            "clearly instead of inventing a comparison. "
        )
    if extra.strip():
        analysis += "Additional user question: " + extra.strip()
    return common + analysis


def build_overlap_tasks(selected_agencies, selected_years, selected_categories,
                        selected_scope, extra):
    """One independent, bounded Supervisor request per category and fiscal year."""
    tasks = []
    for year in sorted(selected_years):
        year_number = int(year.removeprefix("FY"))
        start_date = f"{year_number - 1}-10-01"
        end_date = f"{year_number}-09-30"
        for category in selected_categories:
            prompt = (
                f"Analyze only {category} contract overlap for {year} "
                f"({start_date} through {end_date}) across "
                f"{', '.join(selected_agencies)}. "
                "Call analyze_contract_overlap exactly once for this category and "
                "this fiscal year, with start_date and end_date as specified, "
                "keyword set to the category, analysis_method='both', "
                "pages_per_agency=2, page_size=100, max_candidates=10, "
                f"Set cross_agency_only={str(selected_scope == 'cross').lower()}. "
                "Do not include another fiscal year or category. "
                "The tool's ranked_candidates are preliminary text matches; "
                "only groups passed stricter screening. Never claim preliminary "
                "matches passed the semantic threshold unless their actual scores "
                "demonstrate that. Do not assert duplication, waste or savings. "
                "Return at most 200 words: awards retrieved/analyzed, preliminary "
                "relationships, qualified groups, notable evidence, and coverage "
                "limitations. Do not list individual awards or repeat contract "
                "descriptions; the client displays structured contract details. "
                "Include source URLs when discussing specific evidence. "
            )
            if extra.strip():
                prompt += "Additional user question: " + extra.strip()
            tasks.append({"label": f"{year} · {category.title()}", "prompt": prompt})
    return tasks


overlap_tab, spending_tab = st.tabs(["Contract Overlap", "Spending Comparison"])
with overlap_tab:
    col_title, col_help = st.columns([3, 2])

    with col_title:
        st.markdown("### Contract Overlap")
    with col_help:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 12px;">
                <span
                    title="Are agencies purchasing similar IT services?&#10;
    Which contracts have related scopes and overlapping periods?&#10;
    What does semantic analysis uncover beyond keyword matching?&#10;
    Which contracts warrant further acquisition review?"
                    style="cursor: help; border-bottom: 1px dotted gray;">
                    ⓘ Analysis Description
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write(
        "Find contracts describing similar work, assess overlapping periods "
        "of performance, and surface candidates for acquisition review."
    )

    col_compare, col_analyze = st.columns(
        [5, 1],
        vertical_alignment="center"
    )

    with col_compare:
        col_label, col_options = st.columns(
            [1.2, 8.8],
            gap="small",
            vertical_alignment="center"
        )               

        with col_label:
            st.markdown("**Compare**")

        with col_options:
            overlap_scope = st.radio(
                "Compare",
                [
                    "Across selected agencies",
                    "Within and across selected agencies",
                ],
                horizontal=True,
                label_visibility="collapsed",
                key="overlap_scope",
            )

    with col_analyze:
        run_analysis = st.button(
            "Analyze",
            type="primary",
            disabled=not valid_filters,
            use_container_width=True,
            key="run_overlap",
        )

    with st.expander("Additional question (optional)", expanded=False):
        overlap_extra = st.text_area(
            "Provide additional instructions",
            placeholder="e.g., Focus on cloud security operations or prioritize contracts with overlapping performance periods.",
            key="overlap_extra",
            label_visibility="collapsed",
            height=75,
        )
    if run_analysis:
        st.session_state.selected_prompt = build_analysis_question(
            "overlap",
            "cross" if overlap_scope == "Across selected agencies" else "all",
            overlap_extra,
        )
        st.session_state.last_overlap_prompt = st.session_state.selected_prompt
        st.rerun()

    if st.session_state.overlap_analyses:
        st.markdown("### Results by fiscal year")
        for label, result in sorted(st.session_state.overlap_analyses.items(), reverse=True):
            with st.container(border=True):
                st.markdown(f"### {label}")
                render_overlap_dashboard(result)
        missing = [label for label in st.session_state.overlap_expected
                   if label not in st.session_state.overlap_analyses]
        if missing and not st.session_state.overlap_queue and st.session_state.pending_response is None:
            st.warning("No structured analysis was returned for: " + ", ".join(missing))
    elif (st.session_state.overlap_expected and not st.session_state.overlap_queue
          and st.session_state.pending_response is None):
        st.info("No structured overlap results have been received for this request yet.")

with spending_tab:
    st.subheader("Compare federal agency spending")
    st.write(
        "Explore spending patterns and acquisition activity across agencies "
        "and IT service categories."
    )
    st.markdown(
        "**Questions this analysis can help answer**\n"
        "- How does reported IT contracting activity compare across agencies?\n"
        "- Which categories show changes over selected fiscal years?\n"
        "- How concentrated are awards among vendors?\n"
        "- Which patterns warrant deeper investigation?"
    )
    st.info(
        "Spending comparison is an exploratory capability. Current MCP tools "
        "may not provide complete, normalized agency spending metrics."
    )
    spending_extra = st.text_area(
        "Additional question (optional)",
        placeholder="e.g., Compare vendor concentration in cybersecurity.",
        key="spending_extra",
        height=75,
    )
    if st.button("Analyze", type="primary",
                 disabled=not valid_filters, key="run_spending"):
        st.session_state.selected_prompt = build_analysis_question(
            "spending", extra=spending_extra,
        )
        st.rerun()

_, col_clear = st.columns([8, 1])

with col_clear:
    if st.button(
        "Clear Chat",
        use_container_width=True,
        key="clear_chat",
    ):
        st.session_state.messages = []
        st.session_state.conversation_input = []
        st.session_state.pending_response = None
        st.session_state.pending_input = None
        st.session_state.overlap_analyses = {}
        st.session_state.overlap_expected = []
        st.session_state.overlap_queue = []
        st.session_state.overlap_active = None
        st.session_state.pop("selected_prompt", None)
        st.session_state.pop("last_overlap_prompt", None)
        st.rerun()

# ---------------------------------------------------------
# Conversation history
# ---------------------------------------------------------

for message in st.session_state.messages:
    if message["role"] == "assistant" and message.get("is_overlap_report"):
        with st.expander("AI analysis and supporting explanation", expanded=False):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# ---------------------------------------------------------
# Pending MCP approval
# ---------------------------------------------------------

if st.session_state.pending_response is not None:

    approvals = get_approval_requests(
        st.session_state.pending_response
    )

    if approvals:

        st.warning(
            "The Supervisor Agent is requesting permission "
            "to execute MCP tools."
        )

        for request in approvals:

            st.markdown(f"**Tool:** `{request.name}`")

            with st.expander("View tool arguments"):
                st.code(
                    str(request.arguments),
                    language="json",
                )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "✅ Approve",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Executing approved MCP tools..."
                    ):
                        resolve_approval(True)

                    st.rerun()

                except Exception as exc:
                    st.error(
                        "The MCP tool execution could not "
                        "be completed."
                    )
                    st.exception(exc)

        with col2:
            if st.button(
                "❌ Deny",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Sending denial to Supervisor..."
                    ):
                        resolve_approval(False)

                    st.rerun()

                except Exception as exc:
                    st.error(
                        "Could not process the denial."
                    )
                    st.exception(exc)


# ---------------------------------------------------------
# New user question
# ---------------------------------------------------------

else:

    question = st.chat_input(
        "Ask about federal contracts, spending, or overlaps..."
    )

    if "selected_prompt" in st.session_state:
        question = st.session_state.pop("selected_prompt")

    is_generated_overlap = bool(question) and question == st.session_state.get("last_overlap_prompt")
    if is_generated_overlap:
        st.session_state.overlap_analyses = {}
        st.session_state.overlap_expected = [
            f"{year} · {category.title()}"
            for year in fiscal_years for category in categories
        ]
        st.session_state.overlap_queue = build_overlap_tasks(
            list(agencies), list(fiscal_years), list(categories),
            "cross" if overlap_scope == "Across selected agencies" else "all",
            overlap_extra,
        )
        st.session_state.overlap_active = None
        display_question = (
            f"Analyze {', '.join(categories)} contracts across {', '.join(agencies)} "
            f"for {', '.join(fiscal_years)}."
        )
        st.session_state.messages.append({"role": "user", "content": display_question})
    elif question:
        # Ordinary chat and spending comparison keep their existing history.
        st.session_state.overlap_expected = []
        st.session_state.messages.append({"role": "user", "content": question})

    # Process one task per Streamlit rerun; approval handling remains unchanged.
    if st.session_state.overlap_queue:
        task = st.session_state.overlap_queue.pop(0)
        st.session_state.overlap_active = task["label"]
        submitted_input = [{"role": "user", "content": task["prompt"]}]
        with st.spinner(f"Analyzing {task['label']}..."):
            try:
                response = submit_to_supervisor(submitted_input)
                handle_supervisor_response(response, submitted_input)
                st.rerun()
            except Exception as exc:
                st.session_state.overlap_queue = []
                st.session_state.overlap_active = None
                st.error(f"Analysis failed for {task['label']}.")
                st.exception(exc)
    elif question and not is_generated_overlap:
        try:
            with st.spinner("Analyzing federal spending..."):
                submitted_input = (
                    list(st.session_state.conversation_input)
                    + [{"role": "user", "content": question}]
                )
                response = submit_to_supervisor(submitted_input)
                handle_supervisor_response(response, submitted_input)
            st.rerun()
        except Exception as exc:
            st.error("The Federal Spending Assistant encountered an error.")
            st.exception(exc)

st.caption(
    "Data source: USAspending.gov | "
    "AI-generated findings identify potential contract relationships "
    "and do not establish confirmed duplication, waste, or savings."
)