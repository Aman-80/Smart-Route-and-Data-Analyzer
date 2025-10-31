"""
Smart Route and Data Analyzer
We are using things to complete this project:---
1 Tkinter GUI
2 Dijkstra shortest path
3 Quick Sort and Merge Sort (custom implementations)
4 SQLite persistence
5 Matplotlib visualizations embedded in Tkinter
6 Modularity via functions and classes inside one file for easy running
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sqlite3
import json
import time
import threading
import random
import os
from datetime import datetime

# numpy import
try:
    import numpy as np
except Exception:
    np = None

# Matplotlib for embedding
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg




# Database helpers (SQLite)----
DB_PATH = os.path.join(os.path.dirname(__file__), 'smart_route_data.db')


def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS graphs (
                    id INTEGER PRIMARY KEY, name TEXT, created_at TEXT,
                    nodes_json TEXT, edges_json TEXT)
                """)
    cur.execute("""CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY, name TEXT, created_at TEXT, data_blob TEXT)
                """)
    cur.execute("""CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY, type TEXT, params TEXT, output TEXT, runtime_ms REAL, created_at TEXT)
                """)
    conn.commit()
    conn.close()


def save_graph_to_db(name, nodes, edges, path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("INSERT INTO graphs (name, created_at, nodes_json, edges_json) VALUES (?,?,?,?)",
                (name, datetime.utcnow().isoformat(), json.dumps(nodes), json.dumps(edges)))
    conn.commit()
    conn.close()


def load_graphs_from_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT id, name, created_at, nodes_json, edges_json FROM graphs ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    graphs = []
    for r in rows:
        gid, name, created_at, nodes_json, edges_json = r
        try:
            nodes = json.loads(nodes_json)
            edges = json.loads(edges_json)
        except Exception:
            nodes, edges = {}, []
        graphs.append({'id': gid, 'name': name, 'created_at': created_at, 'nodes': nodes, 'edges': edges})
    return graphs


def save_dataset_to_db(name, data_list, path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("INSERT INTO datasets (name, created_at, data_blob) VALUES (?,?,?)",
                (name, datetime.utcnow().isoformat(), json.dumps(data_list)))
    conn.commit()
    conn.close()


def load_datasets_from_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT id, name, created_at, data_blob FROM datasets ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    datasets = []
    for r in rows:
        did, name, created_at, data_blob = r
        try:
            data = json.loads(data_blob)
        except Exception:
            data = []
        datasets.append({'id': did, 'name': name, 'created_at': created_at, 'data': data})
    return datasets


def save_result_to_db(type_, params, output, runtime_ms, path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("INSERT INTO results (type, params, output, runtime_ms, created_at) VALUES (?,?,?,?,?)",
                (type_, json.dumps(params), json.dumps(output), runtime_ms, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()





# Algorithm implementations
# Dijkstra's algorithm for adjacency dict: {node: [(neighbor, weight), ...], ...}
import heapq


def dijkstra(adj, source, target):
    if source not in adj or target not in adj:
        return None, []
    dist = {node: float('inf') for node in adj}
    prev = {node: None for node in adj}
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == target:
            break
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dist[target] == float('inf'):
        return None, []
    # reconstruct path
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return dist[target], path


# Quick Sort - returns new list

def quicksort(arr):
    a = list(arr)

    def _qs(a, lo, hi):
        if lo < hi:
            p = partition(a, lo, hi)
            _qs(a, lo, p - 1)
            _qs(a, p + 1, hi)

    def partition(a, lo, hi):
        pivot = a[hi]
        i = lo - 1
        for j in range(lo, hi):
            if a[j] <= pivot:
                i += 1
                a[i], a[j] = a[j], a[i]
        a[i + 1], a[hi] = a[hi], a[i + 1]
        return i + 1

    _qs(a, 0, len(a) - 1)
    return a


# Merge Sort - returns new list

def mergesort(arr):
    a = list(arr)
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left = mergesort(a[:mid])
    right = mergesort(a[mid:])
    return merge(left, right)


def merge(left, right):
    i = j = 0
    out = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out


# Timing complexity

def time_function(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, out


# Visualization helpers
def draw_graph_on_axes(ax, nodes_dict, edges_list, path_nodes=None):
    """nodes_dict: {id:(x,y)} ; edges_list: [(u,v,weight), ...]"""
    ax.clear()
    # draw edges
    for (u, v, w) in edges_list:
        if u in nodes_dict and v in nodes_dict:
            x1, y1 = nodes_dict[u]
            x2, y2 = nodes_dict[v]
            ax.plot([x1, x2], [y1, y2])
            # mid-point weight label
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax.text(mx, my, f"{w}", fontsize=8)
    # draw nodes
    for nid, (x, y) in nodes_dict.items():
        ax.scatter(x, y)
        ax.text(x, y, str(nid), fontsize=10, weight='bold')
    # draw path if present
    if path_nodes and len(path_nodes) >= 2:
        xs = [nodes_dict[n][0] for n in path_nodes]
        ys = [nodes_dict[n][1] for n in path_nodes]
        ax.plot(xs, ys, linewidth=3, linestyle='-', marker='o')
    ax.set_title('Graph Visualization')
    ax.axis('equal')
    ax.figure.canvas.draw()


def plot_sort_times_on_axes(ax, sizes, times_qs, times_ms, times_py=None):
    ax.clear()
    ax.plot(sizes, times_qs, label='QuickSort')
    ax.plot(sizes, times_ms, label='MergeSort')
    if times_py is not None:
        ax.plot(sizes, times_py, label='Python sorted()')
    ax.set_xlabel('Dataset size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Sorting Performance')
    ax.legend()
    ax.figure.canvas.draw()


# Utilities: parsing graph, sample data
def build_adj_from_edges(edges):
    nodes = set()
    adj = {}
    for u, v, w in edges:
        nodes.add(u); nodes.add(v)
        adj.setdefault(u, []).append((v, float(w)))
        adj.setdefault(v, []).append((u, float(w)))  # undirected
    # ensure nodes with no edges appear
    for n in nodes:
        adj.setdefault(n, [])
    return adj


def sample_graph():
    # small example graph with coordinates
    nodes = {'A': (0, 0), 'B': (2, 1), 'C': (4, 0), 'D': (4, 3), 'E': (1, 4)}
    edges = [
        ('A', 'B', 2.2), ('A', 'E', 4.0), ('B', 'C', 2.5),
        ('B', 'E', 2.0), ('C', 'D', 3.0), ('E', 'D', 3.5)
    ]
    return nodes, edges


def random_dataset(n, use_numpy=False):
    if use_numpy and np is not None:
        return np.random.randint(0, 100000, size=n).tolist()
    else:
        return [random.randint(0, 100000) for _ in range(n)]

# GUI Application
class SmartRouteApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Smart Route and Data Analyzer')
        self.geometry('1100x700')
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        # initialize DB
        init_db()

        # Application state
        self.nodes = {}
        self.edges = []
        self.current_adj = {}
        self.current_dataset = []

        # Setup UI
        self.create_widgets()

    def create_widgets(self):
        # left control frame
        left = ttk.Frame(self, width=320)
        left.pack(side='left', fill='y', padx=8, pady=8)

        # --- Graph controls ---
        gbox = ttk.LabelFrame(left, text='Graph / Dijkstra')
        gbox.pack(fill='x', pady=4)

        ttk.Button(gbox, text='Load Sample Graph', command=self.load_sample_graph).pack(fill='x', pady=2)
        ttk.Button(gbox, text='Save Graph to DB', command=self.save_graph_db).pack(fill='x', pady=2)
        ttk.Button(gbox, text='Load Graphs from DB', command=self.load_graphs_list).pack(fill='x', pady=2)

        ttk.Label(gbox, text='Source Node:').pack(anchor='w', padx=4)
        self.src_var = tk.StringVar()
        self.src_cb = ttk.Combobox(gbox, textvariable=self.src_var, state='readonly')
        self.src_cb.pack(fill='x', padx=4, pady=2)

        ttk.Label(gbox, text='Target Node:').pack(anchor='w', padx=4)
        self.tgt_var = tk.StringVar()
        self.tgt_cb = ttk.Combobox(gbox, textvariable=self.tgt_var, state='readonly')
        self.tgt_cb.pack(fill='x', padx=4, pady=2)

        ttk.Button(gbox, text='Run Dijkstra', command=self.run_dijkstra_background).pack(fill='x', pady=4)

        # Sorting controls---
        sbox = ttk.LabelFrame(left, text='Sorting')
        sbox.pack(fill='x', pady=6)

        ttk.Button(sbox, text='Generate Random Dataset (1000)', command=self.gen_dataset_1000).pack(fill='x', pady=2)
        ttk.Button(sbox, text='Save Dataset to DB', command=self.save_dataset_db).pack(fill='x', pady=2)
        ttk.Button(sbox, text='Load Datasets from DB', command=self.load_datasets_list).pack(fill='x', pady=2)

        ttk.Label(sbox, text='Choose Sort:').pack(anchor='w', padx=4)
        self.sort_choice = tk.StringVar(value='quicksort')
        ttk.Radiobutton(sbox, text='Quick Sort', variable=self.sort_choice, value='quicksort').pack(anchor='w', padx=8)
        ttk.Radiobutton(sbox, text='Merge Sort', variable=self.sort_choice, value='mergesort').pack(anchor='w', padx=8)
        ttk.Radiobutton(sbox, text="Python's sorted()", variable=self.sort_choice, value='python').pack(anchor='w', padx=8)

        ttk.Button(sbox, text='Run Sort & Show Time', command=self.run_sort_background).pack(fill='x', pady=6)
        ttk.Button(sbox, text='Compare Sorts (sizes)', command=self.compare_sorts_background).pack(fill='x', pady=2)

        #Export / Logs ---
        exbox = ttk.LabelFrame(left, text='Export / Logs')
        exbox.pack(fill='both', expand=True, pady=6)

        ttk.Button(exbox, text='Export Current Path to CSV', command=self.export_path).pack(fill='x', pady=2)
        ttk.Button(exbox, text='Export Current Dataset to CSV', command=self.export_dataset).pack(fill='x', pady=2)

        ttk.Label(exbox, text='Logs:').pack(anchor='w')
        self.log = scrolledtext.ScrolledText(exbox, height=10)
        self.log.pack(fill='both', expand=True, padx=4, pady=4)

        # right visualization frame
        right = ttk.Frame(self)
        right.pack(side='right', fill='both', expand=True)

        # Matplotlib figure with two subplots
        self.fig = Figure(figsize=(7, 6))
        self.ax_graph = self.fig.add_subplot(211)
        self.ax_perf = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # bottom status
        self.status = ttk.Label(self, text='Ready', relief='sunken', anchor='w')
        self.status.pack(side='bottom', fill='x')

    #Graph functions---
    def load_sample_graph(self):
        self.nodes, self.edges = sample_graph()
        self.current_adj = build_adj_from_edges(self.edges)
        self.update_node_comboboxes()
        draw_graph_on_axes(self.ax_graph, self.nodes, self.edges, None)
        self.log_insert('Sample graph loaded')

    def save_graph_db(self):
        if not self.nodes or not self.edges:
            messagebox.showwarning('No graph', 'No graph loaded to save')
            return
        name = f"graph_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        save_graph_to_db(name, self.nodes, self.edges)
        self.log_insert(f'Graph saved to DB as {name}')

    def load_graphs_list(self):
        graphs = load_graphs_from_db()
        if not graphs:
            messagebox.showinfo('No graphs', 'No saved graphs found')
            return
        # simple selection: pick the first (most recent)
        g = graphs[0]
        self.nodes = g['nodes']
        self.edges = g['edges']
        self.current_adj = build_adj_from_edges(self.edges)
        self.update_node_comboboxes()
        draw_graph_on_axes(self.ax_graph, self.nodes, self.edges, None)
        self.log_insert(f"Loaded graph '{g['name']}' from DB")

    def update_node_comboboxes(self):
        nodes_list = sorted(list(self.current_adj.keys()))
        self.src_cb['values'] = nodes_list
        self.tgt_cb['values'] = nodes_list
        if nodes_list:
            self.src_cb.set(nodes_list[0])
            self.tgt_cb.set(nodes_list[-1])

    def run_dijkstra_background(self):
        t = threading.Thread(target=self.run_dijkstra)
        t.daemon = True
        t.start()

    def run_dijkstra(self):
        src = self.src_var.get()
        tgt = self.tgt_var.get()
        if not src or not tgt:
            messagebox.showwarning('Missing nodes', 'Select source and target nodes')
            return
        self.set_status('Running Dijkstra...')
        ms, out = time_function(dijkstra, self.current_adj, src, tgt)
        dist, path = out
        if dist is None:
            self.log_insert(f'No path between {src} and {tgt}')
            self.set_status('No path found')
        else:
            save_result_to_db('dijkstra', {'source': src, 'target': tgt}, {'distance': dist, 'path': path}, ms)
            self.log_insert(f'Dijkstra: distance={dist:.3f}, path={path}, time={ms:.2f} ms')
            # overlay path on graph
            draw_graph_on_axes(self.ax_graph, self.nodes, self.edges, path)
            self.set_status(f'Dijkstra done in {ms:.2f} ms')

    #Dataset functions-----
    def gen_dataset_1000(self):
        self.current_dataset = random_dataset(1000, use_numpy=(np is not None))
        self.log_insert('Generated random dataset of size 1000')

    def save_dataset_db(self):
        if not self.current_dataset:
            messagebox.showwarning('No data', 'No dataset to save')
            return
        name = f"dataset_{len(self.current_dataset)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        save_dataset_to_db(name, self.current_dataset)
        self.log_insert(f'Dataset saved to DB as {name}')

    def load_datasets_list(self):
        ds = load_datasets_from_db()
        if not ds:
            messagebox.showinfo('No datasets', 'No saved datasets found')
            return
        d0 = ds[0]
        self.current_dataset = d0['data']
        self.log_insert(f"Loaded dataset '{d0['name']}' (size={len(self.current_dataset)})")

    def run_sort_background(self):
        t = threading.Thread(target=self.run_sort)
        t.daemon = True
        t.start()

    def run_sort(self):
        if not self.current_dataset:
            messagebox.showwarning('No data', 'Generate or load a dataset first')
            return
        choice = self.sort_choice.get()
        arr = list(self.current_dataset)
        self.set_status(f'Running {choice}...')
        if choice == 'quicksort':
            ms, out = time_function(quicksort, arr)
        elif choice == 'mergesort':
            ms, out = time_function(mergesort, arr)
        else:
            ms, out = time_function(sorted, arr)
        save_result_to_db('sort', {'algorithm': choice, 'size': len(arr)}, {'sample_output': out[:20]}, ms)
        self.log_insert(f'{choice} finished in {ms:.2f} ms (sample first 20: {out[:20]})')
        self.set_status(f'Sort done in {ms:.2f} ms')

    def compare_sorts_background(self):
        t = threading.Thread(target=self.compare_sorts)
        t.daemon = True
        t.start()

    def compare_sorts(self):
        # compare across sizes and plot
        sizes = [100, 500, 1000, 3000]
        times_qs = []
        times_ms = []
        times_py = []
        self.set_status('Comparing sorts...')
        for n in sizes:
            data = random_dataset(n, use_numpy=(np is not None))
            ms_qs, _ = time_function(quicksort, data)
            ms_ms, _ = time_function(mergesort, data)
            ms_py, _ = time_function(sorted, data)
            times_qs.append(ms_qs)
            times_ms.append(ms_ms)
            times_py.append(ms_py)
            self.log_insert(f'size={n}: QS={ms_qs:.2f} ms, MS={ms_ms:.2f} ms, PY={ms_py:.2f} ms')
        plot_sort_times_on_axes(self.ax_perf, sizes, times_qs, times_ms, times_py)
        self.set_status('Sort comparison done')

    #Export helpers-----
    def export_path(self):
        # find last saved dijkstra result from DB
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT params, output FROM results WHERE type='dijkstra' ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if not row:
            messagebox.showinfo('No path', 'No dijkstra result found in DB to export')
            return
        params_json, output_json = row
        params = json.loads(params_json)
        output = json.loads(output_json)
        path = output.get('path', [])
        if not path:
            messagebox.showinfo('No path', 'Last dijkstra result had no path')
            return
        f = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not f:
            return
        with open(f, 'w') as fh:
            fh.write('node\n')
            for node in path:
                fh.write(f"{node}\n")
        self.log_insert(f'Exported path to {f}')

    def export_dataset(self):
        if not self.current_dataset:
            messagebox.showwarning('No dataset', 'No dataset to export')
            return
        f = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not f:
            return
        with open(f, 'w') as fh:
            for v in self.current_dataset:
                fh.write(f"{v}\n")
        self.log_insert(f'Exported dataset to {f}')

    #helpers ----
    def log_insert(self, text):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log.insert('end', f'[{ts}] {text}\n')
        self.log.see('end')

    def set_status(self, text):
        self.status.config(text=text)

    def on_close(self):
        self.destroy()

# Entry point
if __name__ == '__main__':
    app = SmartRouteApp()
    app.mainloop()
