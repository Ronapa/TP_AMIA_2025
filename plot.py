import matplotlib.pyplot as plt
import numpy as np


def plot_benchmark_results(results, graphs=None):
    """
    Genera gráficos de benchmarking para múltiples datasets y modelos.

    Parameters
    ----------
    results : dict
        Diccionario con resultados de benchmarking por dataset.
        Cada valor debe tener un método .summary(baseline='QDA').
    graphs : dict, optional
        Diccionario que controla qué gráficos mostrar. Keys:
        - 'train_time': Tiempo de entrenamiento
        - 'train_speedup': Speedup de entrenamiento
        - 'test_time': Tiempo de test
        - 'speedup': Speedup de test
        - 'memory': Memoria de test
        Si es None, muestra todos los gráficos.

    Returns
    -------
    fig : matplotlib.figure.Figure
        La figura generada.
    """
    # Configuración por defecto: mostrar todos los gráficos
    if graphs is None:
        graphs = {
            'train_time': True,
            'train_speedup': True,
            'test_time': True,
            'speedup': True,
            'memory': True
        }

    # Determinar qué gráficos mostrar
    active_graphs = [key for key, show in graphs.items() if show]
    n_graphs = len(active_graphs)

    if n_graphs == 0:
        raise ValueError("Al menos un gráfico debe estar habilitado en el diccionario 'graphs'")

    # Configuración de gráficos
    graph_config = {
        'train_time': {
            'data_key': 'train_median_ms',
            'label': 'Train Time (ms)',
            'title': 'Training Time',
            'color': 'steelblue',
            'log_scale': False,
            'value_format': '{:.2f}',
            'is_speedup': False
        },
        'train_speedup': {
            'data_key': 'train_speedup',
            'label': 'Train Speedup',
            'title': 'Train Speedup',
            'color': 'steelblue',
            'log_scale': False,
            'value_format': '{:.1f}x',
            'is_speedup': True
        },
        'test_time': {
            'data_key': 'test_median_ms',
            'label': 'Test Time (ms) - log scale',
            'title': 'Test Time',
            'color': 'coral',
            'log_scale': True,
            'value_format': '{:.2f}',
            'is_speedup': False
        },
        'speedup': {
            'data_key': 'test_speedup',
            'label': 'Test Speedup',
            'title': 'Test Speedup',
            'color': 'coral',
            'log_scale': False,
            'value_format': '{:.1f}x',
            'is_speedup': True
        },
        'memory': {
            'data_key': 'test_mem_median_mb',
            'label': 'Test Memory (MB) - log scale',
            'title': 'Test Memory',
            'color': 'green',
            'log_scale': True,
            'value_format': '{:.2f}',
            'is_speedup': False
        }
    }

    # Crear datasets
    datasets = list(results.keys())
    n_datasets = len(datasets)

    # Crear figura con subplots
    fig, axes = plt.subplots(n_datasets, n_graphs, figsize=(5 * n_graphs, 4 * n_datasets))

    # Asegurar que axes sea 2D
    if n_datasets == 1 and n_graphs == 1:
        axes = np.array([[axes]])
    elif n_datasets == 1:
        axes = axes.reshape(1, -1)
    elif n_graphs == 1:
        axes = axes.reshape(-1, 1)

    # Graficar cada dataset
    for row_idx, dataset_name in enumerate(datasets):
        bench = results[dataset_name]

        # Obtener summary con baseline
        summary = bench.summary(baseline='QDA')

        # Datos generales
        models = summary.index.tolist()
        x = range(len(models))

        # Graficar cada métrica activa
        for col_idx, graph_key in enumerate(active_graphs):
            config = graph_config[graph_key]
            ax = axes[row_idx, col_idx]

            # Obtener datos
            data = summary[config['data_key']].values

            # Crear barras
            bars = ax.bar(x, data, color=config['color'], alpha=0.8)

            # Configurar eje Y
            ax.set_ylabel(config['label'], fontsize=11, color=config['color'])
            ax.tick_params(axis='y', labelcolor=config['color'])

            # Título
            ax.set_title(f'{dataset_name}: {config["title"]}',
                        fontsize=11, fontweight='bold')

            # Configurar eje X
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)

            # Grid
            if config['log_scale']:
                ax.set_yscale('log')
                ax.grid(axis='y', alpha=0.3, which='both')
            else:
                ax.grid(axis='y', alpha=0.3)

            # Línea de referencia para speedup
            if config['is_speedup']:
                ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            # Agregar valores a las barras
            ylim = ax.get_ylim()
            for bar, val in zip(bars, data):
                height = bar.get_height()

                # Calcular posición del texto según escala
                if config['log_scale']:
                    # En escala log, usar media geométrica
                    center_y = np.sqrt(ylim[0] * height)
                else:
                    # En escala lineal, usar punto medio
                    center_y = height / 2

                # Formatear valor
                text = config['value_format'].format(val)

                # Agregar texto
                ax.text(bar.get_x() + bar.get_width() / 2., center_y,
                       text, ha='center', va='center', fontsize=7, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor=config['color'],
                                alpha=0.7,
                                edgecolor='none'))

    plt.tight_layout()
    return fig


# Ejemplo de uso:
if __name__ == '__main__':
    # Configurar qué gráficos mostrar
    graphs = {
        'train_time': True,
        'train_speedup': True,
        'test_time': True,
        'speedup': True,
        'memory': True
    }

    # Llamar la función (asumiendo que 'results' está definido)
    # fig = plot_benchmark_results(results, graphs)
    # plt.show()
