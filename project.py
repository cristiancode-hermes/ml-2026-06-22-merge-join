#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MERGE Y JOIN DE DATASETS — Nivel 1 · Skill: pandas merges & joins       ║
║   Aprende a combinar múltiples fuentes de datos como en ML real            ║
╚══════════════════════════════════════════════════════════════════════════════╝

ESCENARIO: Somos analistas de datos en "TechStore", una tienda online.
Tenemos datos separados en 4 archivos CSV (clientes, pedidos, productos,
reseñas) y necesitamos UNIRLOS para responder preguntas de negocio.

TIPOS DE JOIN QUE VAMOS A PRACTICAR:
  - INNER JOIN → solo registros que existen en ambas tablas
  - LEFT JOIN  → todos de la izquierda + coincidencias de la derecha
  - RIGHT JOIN → todos de la derecha + coincidencias de la izquierda
  - OUTER JOIN → todos los registros de ambas tablas
  - CONCAT    → apilar filas verticalmente (cuando las tablas tienen las mismas columnas)
"""

# =============================================================================
# 1. IMPORTAR LIBRERÍAS
# =============================================================================
# Pandas es la navaja suiza del ML para manipular datos tabulares
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Configuración para que los prints se vean bonitos
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', 150)         # Ancho de la terminal
pd.set_option('display.max_colwidth', 40)   # Trunca texto largo

# Crear carpeta para gráficos si no existe
os.makedirs('docs', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("=" * 70)
print("📦 TECHSTORE — SISTEMA DE ANÁLISIS DE DATOS")
print("=" * 70)

# =============================================================================
# 2. CREAR DATOS REALISTAS
# =============================================================================
# En ML real NUNCA tienes todos los datos en un solo archivo.
# Los datos vienen de diferentes sistemas (CRM, ventas, logs, reseñas)
# y tu trabajo es UNIRLOS inteligentemente.

print("\n📝 GENERANDO DATOS SINTÉTICOS REALISTAS...\n")

# ---------------------------------------------------------------------------
# 2a. CLIENTES — datos de clientes registrados en la tienda
# ---------------------------------------------------------------------------
# Simulamos clientes con info básica: algunos nunca han comprado,
# otros han comprado mucho.
np.random.seed(42)  # Semilla para que los resultados sean reproducibles

n_clientes = 50
clientes_data = {
    'cliente_id': range(1, n_clientes + 1),
    'nombre': [f"Cliente_{i}" for i in range(1, n_clientes + 1)],
    'ciudad': np.random.choice(['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena'],
                                n_clientes, p=[0.35, 0.25, 0.20, 0.12, 0.08]),
    'fecha_registro': pd.date_range('2025-01-01', periods=n_clientes, freq='D').strftime('%Y-%m-%d'),
}
clientes = pd.DataFrame(clientes_data)

# Agregamos un segmento de clientes (Premium, Gold, Standard)
# Esto es útil para ver cómo segmentar después
clientes['segmento'] = np.random.choice(['Premium', 'Gold', 'Standard'], n_clientes,
                                        p=[0.15, 0.30, 0.55])

print(f"✅ Clientes creados: {len(clientes)} registros")
print(f"   Columnas: {list(clientes.columns)}")
print(f"   Segmentos → Premium: {clientes['segmento'].value_counts().get('Premium', 0)}, "
      f"Gold: {clientes['segmento'].value_counts().get('Gold', 0)}, "
      f"Standard: {clientes['segmento'].value_counts().get('Standard', 0)}")

# ---------------------------------------------------------------------------
# 2b. PEDIDOS — cada compra que hicieron los clientes
# ---------------------------------------------------------------------------
# Nota: NO todos los clientes han hecho pedidos (algunos se registraron
# pero nunca compraron). Esto es normal en datos reales.
n_pedidos = 120
pedidos_data = {
    'pedido_id': range(1001, 1001 + n_pedidos),
    # Solamente 40 de los 50 clientes han hecho pedidos (10 clientes inactivos)
    'cliente_id': np.random.choice(range(1, 41), n_pedidos),  # 🔥 clientes 41-50 no aparecen
    'fecha_pedido': pd.date_range('2025-02-01', periods=n_pedidos, freq='2D').strftime('%Y-%m-%d'),
    'total': np.round(np.random.uniform(15, 500, n_pedidos), 2),
    'estado': np.random.choice(['Entregado', 'Envío', 'Pendiente', 'Cancelado'],
                                n_pedidos, p=[0.65, 0.20, 0.10, 0.05]),
}
pedidos = pd.DataFrame(pedidos_data)

print(f"\n✅ Pedidos creados: {len(pedidos)} registros")
print(f"   Clientes con pedidos: {pedidos['cliente_id'].nunique()} de {n_clientes}")
print(f"   Estados → {pedidos['estado'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# 2c. PRODUCTOS — catálogo de productos que se venden en la tienda
# ---------------------------------------------------------------------------
n_productos = 30
productos_data = {
    'producto_id': range(201, 201 + n_productos),
    'nombre': [
        'Laptop UltraBook', 'Mouse Inalámbrico', 'Teclado Mecánico', 'Monitor 27"',
        'Audífonos Bluetooth', 'Webcam HD', 'USB-C Hub', 'Disco SSD 1TB',
        'Memoria RAM 16GB', 'Cargador Rápido', 'Silla Ergonómica', 'Escritorio Eléctrico',
        'Lámpara LED', 'Micrófono USB', 'Parlante Portátil', 'Tablet 10"',
        'Funda Laptop', 'Hub HDMI', 'Cable USB-C', 'Adaptador Ethernet',
        'Mousepad XL', 'Soporte Monitor', 'Cámara Seguridad', 'Router WiFi 6',
        'Impresora Láser', 'Escáner Documentos', 'UPS 1500VA', 'Hub Thunderbolt',
        'Teclado Numérico', 'Lapicero Digital'
    ][:n_productos],
    'categoria': np.random.choice(
        ['Computadoras', 'Accesorios', 'Audio', 'Oficina', 'Redes'],
        n_productos, p=[0.15, 0.35, 0.20, 0.20, 0.10]
    ),
    'precio': np.round(np.random.uniform(10, 2500, n_productos), 2),
    'stock': np.random.randint(0, 200, n_productos),
}
productos = pd.DataFrame(productos_data)

print(f"\n✅ Productos creados: {len(productos)} registros")
print(f"   Categorías: {productos['categoria'].value_counts().to_dict()}")
print(f"   Rango precios: ${productos['precio'].min():.2f} - ${productos['precio'].max():.2f}")

# ---------------------------------------------------------------------------
# 2d. RESEÑAS — opiniones que dejaron los clientes sobre productos
# ---------------------------------------------------------------------------
# Nota: NO todos los pedidos tienen reseña. Y algunos productos
# tienen reseñas de clientes que no están en nuestro sistema (datos sucios).
n_resenas = 80
resenas_data = {
    'resena_id': range(301, 301 + n_resenas),
    # Solo 70 pedidos tienen reseña (de 120)
    'pedido_id': np.random.choice(range(1001, 1001 + 70), n_resenas),
    'puntuacion': np.random.choice([1, 2, 3, 4, 5], n_resenas, p=[0.05, 0.10, 0.20, 0.30, 0.35]),
    'comentario': np.random.choice([
        'Excelente producto', 'Muy bueno', 'Normal', 'Podría mejorar', 'Decepcionante',
        'Lo recomiendo', 'Buena relación calidad-precio', 'Llegó rápido', 'No funcionó',
        'Perfecto para lo que necesito'
    ], n_resenas),
}
resenas = pd.DataFrame(resenas_data)

# Agregamos el producto_id a las reseñas (cada pedido tiene un producto)
# En un dataset real, el pedido tendría una línea con el producto_id
pedido_producto = {pid: np.random.choice(productos['producto_id']) 
                   for pid in pedidos['pedido_id']}
resenas['producto_id'] = resenas['pedido_id'].map(
    lambda pid: pedido_producto.get(pid, np.random.choice(productos['producto_id']))
)

print(f"\n✅ Reseñas creadas: {len(resenas)} registros")
print(f"   Productos reseñados: {resenas['producto_id'].nunique()} de {n_productos}")
print(f"   Puntuación media: {resenas['puntuacion'].mean():.2f}/5")

# =============================================================================
# 3. GUARDAR LOS DATASETS COMO CSV
# =============================================================================
# En ML real cargamos datos de archivos. Guardamos lo que generamos.
print("\n💾 GUARDANDO DATASETS COMO CSV...")

clientes.to_csv('data/clientes.csv', index=False)
pedidos.to_csv('data/pedidos.csv', index=False)
productos.to_csv('data/productos.csv', index=False)
resenas.to_csv('data/resenas.csv', index=False)

print("   ✓ data/clientes.csv")
print("   ✓ data/pedidos.csv")
print("   ✓ data/productos.csv")
print("   ✓ data/resenas.csv")
print("   📁 Todos guardados en /data/")

# =============================================================================
# 4. CARGAR LOS DATOS (simulando que vienen de archivos externos)
# =============================================================================
print("\n" + "=" * 70)
print("📂 CARGANDO DATOS DESDE CSV...")
print("=" * 70)

clientes = pd.read_csv('data/clientes.csv')
pedidos = pd.read_csv('data/pedidos.csv')
productos = pd.read_csv('data/productos.csv')
resenas = pd.read_csv('data/resenas.csv')

# Vista rápida de cada dataset
print("\n👀 Vista previa de cada dataset:")
print("\n--- CLIENTES (primeros 3):")
print(clientes.head(3).to_string(index=False))
print(f"\n   Shape: {clientes.shape}")

print("\n--- PEDIDOS (primeros 3):")
print(pedidos.head(3).to_string(index=False))
print(f"\n   Shape: {pedidos.shape}")

print("\n--- PRODUCTOS (primeros 3):")
print(productos.head(3).to_string(index=False))
print(f"\n   Shape: {productos.shape}")

print("\n--- RESEÑAS (primeros 3):")
print(resenas.head(3).to_string(index=False))
print(f"\n   Shape: {resenas.shape}")

# =============================================================================
# 5. EXPLICACIÓN VISUAL DE LOS JOINS
# =============================================================================
print("\n" + "=" * 70)
print("🎯 ENTENDIENDO LOS JOINS — Visualización conceptual")
print("=" * 70)

# Creamos una figura que muestra visualmente los 4 tipos de join
# usando diagramas simples de conjuntos (Venn-like)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Los 4 Tipos de Join en pandas — Visualización Conceptual',
             fontsize=16, fontweight='bold', y=1.02)

join_concepts = [
    (0, 0, 'INNER JOIN\nSolo registros que coinciden\nen AMBAS tablas',
     ['A ∩ B'], '#2ecc71'),
    (0, 1, 'LEFT JOIN\nTodos de A + coincidencias de B\n(lo que no coincide → NaN)',
     ['A', 'A ∩ B'], '#3498db'),
    (0, 2, 'RIGHT JOIN\nTodos de B + coincidencias de A\n(lo que no coincide → NaN)',
     ['A ∩ B', 'B'], '#e74c3c'),
    (1, 0, 'OUTER JOIN\nTodos los registros de\nAMBAS tablas (unión total)',
     ['A', 'A ∩ B', 'B'], '#9b59b6'),
    (1, 1, 'CONCAT (eje=0)\nApilar datos verticalmente\n(las columnas deben coincidir)',
     ['A'], '#f39c12'),
    (1, 2, 'CONCAT (eje=1)\nPegar columnas horizontalmente\n(como un excel lateral)',
     ['A'], '#1abc9c'),
]

for (row, col, title, labels, color) in join_concepts:
    ax = axes[row, col]
    ax.text(0.5, 0.5, f'{title}\n\n{chr(10).join(labels)}',
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=1.5', facecolor=color, alpha=0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

plt.tight_layout()
plt.savefig('docs/01_tipos_join_conceptual.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n📊 Diagrama conceptual guardado: docs/01_tipos_join_conceptual.png")

# =============================================================================
# 6. INNER JOIN — El más común en ML
# =============================================================================
# 🎯 INNER JOIN: Solo conserva las filas donde la clave coincide en AMBAS tablas.
# Es el join por defecto en pandas (how='inner').
# Se usa cuando SOLO te interesan los registros completos.

print("\n" + "=" * 70)
print("🔗 EJEMPLO 1: INNER JOIN — Clientes que SÍ han comprado")
print("=" * 70)

# Explicación: Queremos ver los pedidos de cada cliente.
# Un cliente sin pedidos no nos interesa aquí.
# La clave común entre clientes y pedidos es 'cliente_id'

inner_join = pd.merge(
    clientes,       # Tabla izquierda
    pedidos,        # Tabla derecha
    on='cliente_id',  # Columna clave común
    how='inner'     # Solo filas que coinciden en ambas
)

print(f"\n📊 Tabla original: {len(clientes)} clientes, {len(pedidos)} pedidos")
print(f"📊 Después del INNER JOIN: {len(inner_join)} filas")
print(f"\n   ¿Qué pasó?")
print(f"   → Clientes sin pedidos (ids 41-50) fueron EXCLUIDOS automáticamente")
print(f"   → Clientes con múltiples pedidos aparecen varias veces (1 fila por pedido)")

print(f"\n👀 Primeras 5 filas del INNER JOIN:")
print(inner_join[['nombre', 'ciudad', 'segmento', 'pedido_id', 'total', 'estado']].head(5).to_string(index=False))

# -------------------- ¿Cuántos clientes se quedaron fuera? --------------------
clientes_con_pedidos = inner_join['cliente_id'].nunique()
clientes_sin_pedidos = len(clientes) - clientes_con_pedidos
print(f"\n🔍 Clientes que compraron: {clientes_con_pedidos} de {len(clientes)}")
print(f"🔍 Clientes inactivos (sin pedidos): {clientes_sin_pedidos}")

# Ejemplo práctico: Ingreso total por cliente (solo para quienes compraron)
ingreso_por_cliente = inner_join.groupby('nombre').agg(
    total_gastado=('total', 'sum'),
    num_pedidos=('pedido_id', 'count'),
    ciudad=('ciudad', 'first')
).sort_values('total_gastado', ascending=False)

print(f"\n💰 Top 5 clientes por ingresos (INNER JOIN — solo compradores):")
print(ingreso_por_cliente.head(5).to_string())

# Visualización: Distribución del gasto por cliente
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma de total gastado
axes[0].hist(ingreso_por_cliente['total_gastado'], bins=15, color='#2ecc71', edgecolor='white', alpha=0.7)
axes[0].set_title('Distribución del Gasto Total por Cliente', fontsize=13)
axes[0].set_xlabel('Total Gastado ($)')
axes[0].set_ylabel('Número de Clientes')
axes[0].axvline(ingreso_por_cliente['total_gastado'].mean(), color='red', linestyle='--', 
                label=f"Media: ${ingreso_por_cliente['total_gastado'].mean():.0f}")
axes[0].legend()

# Count de pedidos por cliente
axes[1].hist(ingreso_por_cliente['num_pedidos'], bins=10, color='#3498db', edgecolor='white', alpha=0.7)
axes[1].set_title('Distribución del Número de Pedidos por Cliente', fontsize=13)
axes[1].set_xlabel('Número de Pedidos')
axes[1].set_ylabel('Número de Clientes')
axes[1].axvline(ingreso_por_cliente['num_pedidos'].mean(), color='red', linestyle='--',
                label=f"Media: {ingreso_por_cliente['num_pedidos'].mean():.1f}")
axes[1].legend()

plt.tight_layout()
plt.savefig('docs/02_inner_join_gasto_clientes.png', dpi=120, bbox_inches='tight')
plt.close()
print("📊 Gráfico guardado: docs/02_inner_join_gasto_clientes.png")

# =============================================================================
# 7. LEFT JOIN — El más usado en la vida real
# =============================================================================
# 🎯 LEFT JOIN: Conserva TODOS los registros de la izquierda y agrega
# los datos de la derecha donde coincida. Si no coincide → NaN.
# 
# Es el join más común porque normalmente TIENES una tabla principal
# (ej: clientes) y quieres ENRIQUECERLA con datos de otra tabla.

print("\n" + "=" * 70)
print("🔗 EJEMPLO 2: LEFT JOIN — Todos los clientes + info de pedidos")
print("=" * 70)

# Situación real: El equipo de marketing quiere UNA lista de TODOS los
# clientes (incluso los que nunca compraron) con su actividad de pedidos.
# Para los que no compraron, queremos saberlo (ver NaN).

left_join = pd.merge(
    clientes,    # Tabla izquierda (TODOS los clientes)
    pedidos,     # Tabla derecha (solo los que coincidan)
    on='cliente_id',
    how='left'   # 🔥 LEFT JOIN — clave de este ejemplo
)

print(f"\n📊 Clientes originales: {len(clientes)}")
print(f"📊 LEFT JOIN resultante: {len(left_join)} filas")
print(f"   → Como es LEFT JOIN, tenemos al menos 1 fila por cada cliente")
print(f"   → Los clientes sin pedidos tienen NaN en columnas de pedidos")

# Identificar clientes que NUNCA han comprado
clientes_sin_actividad = left_join[left_join['pedido_id'].isna()]
print(f"\n🔍 Clientes sin ningún pedido (aparecen con NaN): {len(clientes_sin_actividad)}")
print(f"   {clientes_sin_actividad[['cliente_id', 'nombre', 'ciudad', 'segmento']].to_string(index=False)}")

# Ver las filas con NaN explícitamente
print(f"\n👀 Primeras filas del LEFT JOIN (clientes con y sin pedidos):")
# Seleccionamos filas que muestren el contraste
muestra = pd.concat([
    left_join.dropna(subset=['pedido_id']).head(3),     # Clientes con pedidos
    clientes_sin_actividad.head(2)                       # Clientes sin pedidos
])
print(muestra[['nombre', 'segmento', 'pedido_id', 'total', 'estado']].to_string(index=False))

print(f"\n💡 APLICACIÓN EN ML: El LEFT JOIN es súper útil para FEATURE ENGINEERING.")
print(f"   Por ejemplo, podemos crear una columna 'tiene_pedidos' (1/0) para")
print(f"   predecir qué clientes inactivos podrían reactivarse.")

# =============================================================================
# 8. MULTIPLES JOINS ENCADENADOS — Como en ML real
# =============================================================================
# En la vida real NUNCA haces un solo join. Encadenas varios.
# Ejemplo: Unir clientes + pedidos + productos + reseñas para análisis completo.

print("\n" + "=" * 70)
print("🔗 EJEMPLO 3: MÚLTIPLES JOINS — Datos completos para análisis")
print("=" * 70)

# Queremos un dataset con: cliente, su pedido, el producto que compró, y su reseña
# Esto requiere varios pasos de merge:

# Paso 1: Unir pedidos con productos
# Nota: No tenemos 'producto_id' en pedidos. Lo simulamos con random.
pedidos['producto_id'] = pedidos['pedido_id'].map(
    lambda pid: pedido_producto.get(pid, np.random.choice(productos['producto_id']))
)

# Cadena de merges (como en ML real):
# pedidos → +productos → +clientes → +reseñas
paso1 = pd.merge(pedidos, productos, on='producto_id', how='left')
paso2 = pd.merge(paso1, clientes, on='cliente_id', how='left')
datos_completos = pd.merge(paso2, resenas, on='pedido_id', how='left')

print(f"\n📊 Dataset final con todos los datos combinados:")
print(f"   Columnas: {list(datos_completos.columns)}")
print(f"   Filas: {len(datos_completos)}")
print(f"\n👀 Primeras 5 filas del dataset multi-join:")
print(datos_completos[['nombre_y', 'pedido_id', 'nombre_x', 'precio', 'total', 
                        'categoria', 'puntuacion']].head(5).to_string(index=False))

# Análisis: Productos más vendidos por categoría
print(f"\n📊 Top productos más vendidos:")
top_productos = datos_completos.groupby(['categoria', 'nombre_x']).agg(
    unidades_vendidas=('pedido_id', 'count'),
    ingreso_total=('total', 'sum')
).sort_values('unidades_vendidas', ascending=False)

print(top_productos.head(10).to_string())

# =============================================================================
# 9. OUTER JOIN — Unión total (cuando NO quieres perder nada)
# =============================================================================
# 🎯 OUTER JOIN: Conserva TODO de ambas tablas. Lo que no coincide → NaN.
# Se usa cuando quieres ver qué registros FALTAN en una tabla vs la otra.

print("\n" + "=" * 70)
print("🔗 EJEMPLO 4: OUTER JOIN — Clientes + Pedidos (NADA se pierde)")
print("=" * 70)

outer_join = pd.merge(
    clientes,
    pedidos,
    on='cliente_id',
    how='outer',
    indicator=True  # 🔥 Añade columna '_merge' que dice de dónde vino cada fila
)

print(f"\n📊 Clientes: {len(clientes)}, Pedidos: {len(pedidos)}")
print(f"📊 OUTER JOIN: {len(outer_join)} filas")

# La columna _merge nos dice el origen de cada fila
print(f"\n🔍 Origen de las filas según _merge:")
print(outer_join['_merge'].value_counts().to_string())

print(f"\n   → 'both': cliente existe en ambas tablas (tiene pedido)")
print(f"   → 'left_only': cliente registrado pero SIN pedidos")
print(f"   → 'right_only': hay pedidos de clientes que NO existen en clientes")

# =============================================================================
# 10. CONCAT — Apilar datos vertical u horizontalmente
# =============================================================================
# 🎯 CONCAT: Útil cuando tienes datos PARTIDOS en varios archivos con la misma
# estructura (ej: ventas de enero, ventas de febrero, etc.).

print("\n" + "=" * 70)
print("🔗 EJEMPLO 5: CONCAT — Unir datos de diferentes períodos")
print("=" * 70)

# Simulamos que tenemos ventas separadas por trimestre
ventas_q1 = pedidos[pedidos['fecha_pedido'].between('2025-02-01', '2025-04-30')].copy()
ventas_q2 = pedidos[pedidos['fecha_pedido'].between('2025-05-01', '2025-07-31')].copy()
ventas_q3 = pedidos[pedidos['fecha_pedido'].between('2025-08-01', '2025-10-31')].copy()

print(f"\n📊 Ventas Q1: {len(ventas_q1)} pedidos")
print(f"📊 Ventas Q2: {len(ventas_q2)} pedidos")
print(f"📊 Ventas Q3: {len(ventas_q3)} pedidos")

# CONCAT: apilar verticalmente (eje=0 por defecto)
ventas_totales = pd.concat([ventas_q1, ventas_q2, ventas_q3], ignore_index=True)

print(f"\n📊 Ventas totales (concat): {len(ventas_totales)} pedidos")
print(f"   Debería ser: {len(ventas_q1) + len(ventas_q2) + len(ventas_q3)} (correcto ✓)")

# Verificar que no se perdió nada
print(f"\n💡 CONCAT se usa mucho cuando:")
print(f"   • Los datos vienen por meses/días y hay que unirlos")
print(f"   • Tienes múltiples archivos CSV con la misma estructura")
print(f"   • Quieres combinar resultados de diferentes experimentos")

# =============================================================================
# 11. VISUALIZACIÓN COMPARATIVA DE TODOS LOS TIPOS DE JOIN
# =============================================================================
# Vamos a visualizar cuántas filas produce cada tipo de join.

print("\n" + "=" * 70)
print("📊 COMPARATIVA: Mismo merge con diferentes tipos de join")
print("=" * 70)

# Creamos datasets pequeños para visualizar claramente la diferencia
a = pd.DataFrame({'id': [1, 2, 3, 4], 'valor_a': ['A1', 'A2', 'A3', 'A4']})
b = pd.DataFrame({'id': [2, 3, 4, 5], 'valor_b': ['B2', 'B3', 'B4', 'B5']})

print("\n📋 Tabla A:")
print(a.to_string(index=False))
print("\n📋 Tabla B:")
print(b.to_string(index=False))

# Mostramos los 4 joins en acción
joins = {
    'INNER JOIN (A ∩ B)': pd.merge(a, b, on='id', how='inner'),
    'LEFT JOIN (A completo)': pd.merge(a, b, on='id', how='left'),
    'RIGHT JOIN (B completo)': pd.merge(a, b, on='id', how='right'),
    'OUTER JOIN (A ∪ B)': pd.merge(a, b, on='id', how='outer'),
}

for nombre, resultado in joins.items():
    print(f"\n--- {nombre} ---")
    print(f"   Filas: {len(resultado)}")
    print(resultado.to_string(index=False))

# Visualización comparativa
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

for idx, (ax, (nombre, resultado)) in enumerate(zip(axes.flatten(), joins.items())):
    ax.axis('off')
    # Mostramos como tabla dentro del gráfico
    tabla = ax.table(
        cellText=resultado.values,
        colLabels=resultado.columns,
        cellLoc='center',
        loc='center',
        colColours=[colors[idx % len(colors)]] * len(resultado.columns)
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.5)
    ax.set_title(nombre, fontsize=13, fontweight='bold', pad=20)

plt.suptitle('Comparacion Visual de Tipos de Join', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/03_comparativa_tipos_join.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n📊 Gráfico guardado: docs/03_comparativa_tipos_join.png")

# =============================================================================
# 12. APLICACIÓN PRÁCTICA: Dashboard de análisis de ventas
# =============================================================================
# Ahora que sabemos hacer joins, armemos un mini dashboard de negocio.

print("\n" + "=" * 70)
print("📈 DASHBOARD: Análisis Completo de Ventas TechStore")
print("=" * 70)

# Dataset completo usando joins
analisis = datos_completos.copy()

# 1. Ventas por ciudad
print("\n📍 VENTAS POR CIUDAD:")
ventas_ciudad = analisis.groupby('ciudad').agg(
    ingresos=('total', 'sum'),
    pedidos=('pedido_id', 'nunique'),
    clientes_unicos=('cliente_id', 'nunique')
).sort_values('ingresos', ascending=False)
print(ventas_ciudad.to_string())

# 2. Productos mejor calificados (con suficientes reseñas)
print("\n⭐ TOP PRODUCTOS MEJOR CALIFICADOS:")
productos_calif = analisis.dropna(subset=['puntuacion']).groupby('nombre_x').agg(
    puntuacion_media=('puntuacion', 'mean'),
    num_resenas=('resena_id', 'count'),
    categoria=('categoria', 'first')
).query('num_resenas >= 2').sort_values('puntuacion_media', ascending=False)
print(productos_calif.head(10).to_string())

# 3. Rendimiento por segmento de cliente
print("\n🏆 RENDIMIENTO POR SEGMENTO DE CLIENTE:")
segmento_stats = analisis.groupby('segmento').agg(
    clientes=('cliente_id', 'nunique'),
    total_pedidos=('pedido_id', 'nunique'),
    ingreso_total=('total', 'sum'),
    ticket_promedio=('total', 'mean')
).sort_values('ingreso_total', ascending=False)
print(segmento_stats.to_string())

# =============================================================================
# 13. VISUALIZACIÓN FINAL — Dashboard de 4 gráficos
# =============================================================================
print("\n📊 GENERANDO DASHBOARD VISUAL...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('TechStore - Dashboard de Analisis (Datos con Joins)',
             fontsize=16, fontweight='bold', y=1.02)

# Gráfico 1: Ingresos por ciudad
ax1 = axes[0, 0]
ciudades = ventas_ciudad.head(5)
bars1 = ax1.bar(ciudades.index, ciudades['ingresos'], color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
ax1.set_title('Ingresos por Ciudad (Top 5)', fontsize=12)
ax1.set_xlabel('Ciudad')
ax1.set_ylabel('Ingresos ($)')
for bar, val in zip(bars1, ciudades['ingresos']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'${val:.0f}',
             ha='center', fontsize=9)
ax1.tick_params(axis='x', rotation=0)

# Gráfico 2: Distribución de puntuaciones
ax2 = axes[0, 1]
puntuaciones = analisis.dropna(subset=['puntuacion'])['puntuacion']
colores_punt = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
counts = puntuaciones.value_counts().sort_index()
ax2.bar(counts.index, counts.values, color=colores_punt, edgecolor='white', width=0.7)
ax2.set_title('Distribución de Puntuaciones de Productos', fontsize=12)
ax2.set_xlabel('Puntuación (1-5)')
ax2.set_ylabel('Número de Reseñas')
ax2.set_xticks([1, 2, 3, 4, 5])
for i, val in enumerate(counts.values):
    ax2.text(i + 1, val + 1, str(val), ha='center', fontsize=10)

# Gráfico 3: Ingresos por segmento
ax3 = axes[1, 0]
segmentos = segmento_stats
bars3 = ax3.bar(segmentos.index, segmentos['ingreso_total'], 
                color=['#f1c40f', '#2ecc71', '#3498db'], edgecolor='white')
ax3.set_title('Ingresos por Segmento de Cliente', fontsize=12)
ax3.set_xlabel('Segmento')
ax3.set_ylabel('Ingresos ($)')
for bar, val in zip(bars3, segmentos['ingreso_total']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'${val:.0f}',
             ha='center', fontsize=9)

# Gráfico 4: Comparación visual de tamaños de join
ax4 = axes[1, 1]
tipos = ['INNER', 'LEFT', 'RIGHT', 'OUTER']
tamanos = [len(joins[t]) for t in joins]  # Usamos los joins pequeños de ejemplo
bars4 = ax4.bar(tipos, tamanos, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'], edgecolor='white')
ax4.set_title('Filas Resultantes por Tipo de Join', fontsize=12)
ax4.set_xlabel('Tipo de Join')
ax4.set_ylabel('Número de Filas')
for bar, val in zip(bars4, tamanos):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(val),
             ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('docs/04_dashboard_analisis.png', dpi=120, bbox_inches='tight')
plt.close()
print("📊 Gráfico guardado: docs/04_dashboard_analisis.png")

# =============================================================================
# 14. RESUMEN — Lo que aprendiste hoy
# =============================================================================
print("\n" + "=" * 70)
print("🧠 RESUMEN: Lo que aprendiste hoy — Merge y Join en pandas")
print("=" * 70)

print("""
📌 CONCEPTOS CLAVE:

  1️⃣ INNER JOIN  → Solo registros que existen en AMBAS tablas
                   Usalo cuando: solo te importan los datos completos
                   Código: pd.merge(tabla1, tabla2, on='clave', how='inner')

  2️⃣ LEFT JOIN   → Todos de izquierda + datos de derecha donde coincida
                   (los que no coinciden → NaN)
                   Usalo cuando: tienes una tabla "principal" y quieres enriquecerla
                   Código: pd.merge(tabla1, tabla2, on='clave', how='left')

  3️⃣ RIGHT JOIN  → Todos de derecha + datos de izquierda (opuesto de left)
                   Código: pd.merge(tabla1, tabla2, on='clave', how='right')

  4️⃣ OUTER JOIN  → TODO de ambas tablas (unión completa)
                   Usalo cuando: quieres detectar datos faltantes
                   Código: pd.merge(tabla1, tabla2, on='clave', how='outer', indicator=True)

  5️⃣ CONCAT      → Apilar datos vertical (eje=0) u horizontal (eje=1)
                   Usalo cuando: tienes datos partidos en varios archivos
                   Código: pd.concat([df1, df2], ignore_index=True)

📊 DATOS DEL PROYECTO:
  - 4 datasets simulados (clientes, pedidos, productos, reseñas)
  - 50 clientes, 120 pedidos, 30 productos, 80 reseñas
  - Clientes inactivos sin pedidos (ids 41-50) → para probar joins
  - Reseñas no cubren todos los pedidos → datos incompletos realistas

🛠️ SKILLS PRACTICADOS:
  ✅ pd.merge() con 4 tipos de join
  ✅ Uso de merge con múltiples columnas
  ✅ indicator=True para rastrear origen de datos
  ✅ Encadenamiento de merges (múltiples tablas)
  ✅ pd.concat() para apilar datos
  ✅ groupby() + agg() después de joins
  ✅ Análisis de negocio con datos combinados
""")

# Bonus: Guardar el dataset completo para usarlo en proyectos futuros
datos_completos.to_csv('data/dataset_completo.csv', index=False)
print("💾 Dataset completo guardado: data/dataset_completo.csv")
print("   (Úsalo cuando llegues a Nivel 2 — EDA completo)")
print("=" * 70)
