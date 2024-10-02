import csv
from rdflib import Graph, URIRef, Namespace
import networkx as nx
import plotly.graph_objs as go

def read_txt_file(file_path,g,row_limit):
    # Read triples from your text file
    i = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            i += 1
            if i == row_limit:
                break
            subject, predicate, obj = line.strip().split('\t')  # Split using tab
            g.add((URIRef(subject), URIRef(predicate), URIRef(obj)))

    return g

# Define a base namespace
base_ns = Namespace("http://example.org/")

def to_uri(value):
    # Replace spaces with underscores
    value = value.replace(' ', '_')
    # Replace special characters
    value = value.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    return URIRef(base_ns + value)

def read_csv_file(file_path,g,row_limit):
    # Read triples from the file
    i = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            i += 1
            if i == row_limit:
                break
            subject = to_uri(row['Subject'])
            predicate = to_uri(row['Predicate'])
            obj = to_uri(row['Object'])
            g.add((subject, predicate, obj))

    return g


def export_graph_to_dot_format(g, file_type):
    print("Exporting graph...")
    if file_type == 'txt':
        dot_file_path = 'visualization/wikidata_triples_ids_graph.dot'
        g.serialize(destination=dot_file_path, format='turtle')
        print(f"Graph exported to {dot_file_path}")
    elif file_type == 'csv':
        dot_file_path = 'visualization/wikidata_triples_labels_graph.dot'
        g.serialize(destination=dot_file_path, format='turtle')
        print(f"Graph exported to {dot_file_path}")
    else:
        print('Please choose an appropriate type file')

def convert_rdf_graph(g):
    print("Converting RDF graph...")

    # Convert RDF graph to NetworkX graph
    G = nx.Graph()
    try:
        for s, p, o in g:
            G.add_edge(s, o, label=p)
    except Exception as e:
        print(f"Error processing RDF graph: {e}")
        return None

    # Create a Plotly figure
    pos = nx.spring_layout(G, scale=2)  # Adjust layout as needed
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
    node_trace.marker.color = [G.degree(n) for n in G.nodes()]

    # Create edge label traces with hover text
    edge_label_x = []
    edge_label_y = []
    edge_label_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_label_x.append(mid_x)
        edge_label_y.append(mid_y)
        edge_label_text.append(f'{edge[2]["label"]}')

    edge_label_trace = go.Scatter(
        x=edge_label_x, y=edge_label_y,
        mode='markers',
        hoverinfo='text',
        text=edge_label_text,
        marker=dict(
            size=5,
            color='rgba(255, 0, 0, 0.5)'
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    fig.show()


if __name__ == "__main__":
    g = Graph()
    row_limit = 500 # Use a small value, e.g. for stepwise analysis and to avoid longer computations

    # Read and use an original text file with Wikidata IDs
    file_path_txt_file = '../data/train.txt' # You can choose a different dataset, e.g. test.txt and valid.txt
    g = read_txt_file(file_path_txt_file, g, row_limit)
    export_graph_to_dot_format(g, file_type='txt')
    convert_rdf_graph(g)

    # Optionally, undertake a more human-readable data analysis (Note: added at a later project stage)
    # Read and use a preprocessed text file (see folder 'data_preprocessing' for more info), with Wikidata labels
    #file_path_csv_file = '../data_preprocessing/train_2.0.csv' # Adapt accordingly to original file
    #g = read_csv_file(file_path_csv_file, g, row_limit)
    #export_graph_to_dot_format(g, file_type='csv')
    #convert_rdf_graph(g)