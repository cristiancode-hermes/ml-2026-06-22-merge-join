# 🎯 Merge y Join de Datasets — El arte de unir datos como en ML real

**Nivel:** 1 (Fundamentos Python para ML)
**Skill:** Merge y join de datasets con pandas
**Último proyecto:** 2026-06-21 — Data Cleaning

---

## 📖 Qué vas a aprender

En Machine Learning real, los datos NUNCA vienen en un solo archivo. Imagina que trabajas en una tienda online:

- El **CRM** tiene la info de clientes → `clientes.csv`
- El **sistema de ventas** tiene los pedidos → `pedidos.csv`
- El **catálogo** tiene los productos → `productos.csv`
- El **módulo de reseñas** tiene las opiniones → `resenas.csv`

Tu trabajo como ML Engineer es **COMBINAR** estas fuentes en un solo dataset. Ahí entran los **joins**.

### ¿Qué tipos de join existen?

| Tipo | ¿Qué conserva? | ¿Cuándo usarlo? |
|------|----------------|-----------------|
| **INNER JOIN** | Solo filas que existen en AMBAS tablas | Cuando solo te importan registros completos |
| **LEFT JOIN** | TODAS las filas de la izquierda + datos de la derecha | Cuando tienes una tabla principal y quieres enriquecerla |
| **RIGHT JOIN** | TODAS las filas de la derecha + datos de la izquierda | Lo mismo pero con la tabla derecha como principal |
| **OUTER JOIN** | TODAS las filas de ambas tablas (unión total) | Cuando quieres ver qué falta en cada tabla |
| **CONCAT** | Apila filas vertical u horizontalmente | Datos del mismo tipo partidos en varios archivos |

### Analogía fácil

Imagina dos listas de personas:
- **Lista A:** Los que vinieron a tu fiesta
- **Lista B:** Los que trajeron pastel

- **INNER JOIN** → Los que vinieron Y trajeron pastel
- **LEFT JOIN** → Todos los que vinieron + quiénes trajeron pastel (o NaN si no)
- **RIGHT JOIN** → Todos los que trajeron pastel + si vinieron o no
- **OUTER JOIN** → Todos: los que vinieron, los que trajeron pastel, ¡todos!

---

## 🧪 Código paso a paso

### Paso 1: Importar librerías y crear datos

Primero importamos pandas, numpy, matplotlib y seaborn. Luego generamos **4 datasets realistas** de una tienda online:

```python
# Datos de clientes (50 clientes, algunos nunca compraron)
clientes = pd.DataFrame({
    'cliente_id': range(1, 51),
    'nombre': [f"Cliente_{i}" for i in range(1, 51)],
    'ciudad': np.random.choice(['Bogotá', 'Medellín', 'Cali', ...]),
    'fecha_registro': ...,
    'segmento': np.random.choice(['Premium', 'Gold', 'Standard'])
})

# Datos de pedidos (120 pedidos, solo 40 clientes compraron)
# Los clientes 41-50 no aparecen aquí → datos intencionalmente incompletos
pedidos = pd.DataFrame({...})

# Datos de productos (30 productos)
productos = pd.DataFrame({...})

# Datos de reseñas (80 reseñas, no todos los pedidos tienen reseña)
resenas = pd.DataFrame({...})
```

**🔥 Clave:** Los datos son REALISTAS — hay clientes sin pedidos, pedidos sin reseñas, y productos sin reseñas. Esto pasa SIEMPRE en datos reales.

### Paso 2: INNER JOIN — Solo los que compraron

```python
# INNER JOIN: SOLO clientes que tienen pedidos
inner = pd.merge(clientes, pedidos, on='cliente_id', how='inner')
# Resultado: 120 filas (cada pedido con datos del cliente)
# Los clientes 41-50 (sin pedidos) DESAPARECEN
```

### Paso 3: LEFT JOIN — Todos los clientes (incluso inactivos)

```python
# LEFT JOIN: TODOS los clientes, con info de pedidos donde exista
left = pd.merge(clientes, pedidos, on='cliente_id', how='left')
# Resultado: clientes sin pedidos aparecen con NaN en columnas de pedidos
# PERFECTO para detectar clientes inactivos
```

### Paso 4: Múltiples joins encadenados

```python
# Como en ML real: unimos 3+ tablas en cadena
paso1 = pd.merge(pedidos, productos, on='producto_id', how='left')
paso2 = pd.merge(paso1, clientes, on='cliente_id', how='left')
datos_completos = pd.merge(paso2, resenas, on='pedido_id', how='left')
# ¡Dataset completo con toda la info del negocio!
```

### Paso 5: OUTER JOIN con indicator

```python
# OUTER JOIN: TODO de ambas tablas + columna que muestra el origen
outer = pd.merge(clientes, pedidos, on='cliente_id', how='outer', indicator=True)
# _merge: 'both', 'left_only', 'right_only' → muestra qué existe y qué falta
```

### Paso 6: CONCAT para apilar datos

```python
# Apilar ventas de diferentes trimestres (mismas columnas)
ventas_totales = pd.concat([ventas_q1, ventas_q2, ventas_q3], ignore_index=True)
```

---

## 📊 Resultados

### 📁 Datasets generados
| Dataset | Filas | Columnas | Descripción |
|---------|-------|----------|-------------|
| clientes.csv | 50 | 4 | Info de clientes registrados |
| pedidos.csv | 120 | 5 | Pedidos realizados |
| productos.csv | 30 | 4 | Catálogo de productos |
| resenas.csv | 80 | 4 | Reseñas de productos |

### 📊 Visualizaciones generadas

| Archivo | Descripción |
|---------|-------------|
| `docs/01_tipos_join_conceptual.png` | Diagrama conceptual de los 4 tipos de join + concat |
| `docs/02_inner_join_gasto_clientes.png` | Distribución de gasto por cliente (INNER JOIN) |
| `docs/03_comparativa_tipos_join.png` | Misma tabla unida con 4 tipos de join diferentes |
| `docs/04_dashboard_analisis.png` | Dashboard completo: ingresos por ciudad, puntuaciones, segmentos |

### 🔍 Hallazgos de negocio

```
📍 VENTAS POR CIUDAD:
Bogotá        → $XX,XXX (mayor volumen)
Medellín      → $XX,XXX
...

⭐ TOP PRODUCTOS:
Laptop UltraBook → 4.6/5 (mejor calificado)
...

🏆 RENDIMIENTO POR SEGMENTO:
Premium   → $XX,XXX (mayor ticket promedio)
Gold      → $XX,XXX
Standard  → $XX,XXX
```

---

## 💡 Conceptos clave

- **Merge vs Join:** En pandas, `merge()` es la función principal. `join()` es un atajo para merges por índice. Siempre usa `merge()`.
- **La clave (on):** La columna que conecta las tablas. Debe tener el mismo nombre en ambas (o usa `left_on`/`right_on` si se llaman diferente).
- **how:** Define el tipo de join: `'inner'`, `'left'`, `'right'`, `'outer'`.
- **indicator=True:** Te dice si cada fila vino de la izquierda, la derecha, o ambas. ¡Súper útil para depurar!
- **NaN después de join:** Significa que no había coincidencia en la otra tabla. No es un error, es información.
- **Cardinalidad:** Si después de un join tienes MÁS filas que la tabla original, es porque había duplicados en la clave. ¡Cuidado!

### Errores comunes

❌ Hacer un INNER JOIN cuando necesitas un LEFT JOIN → pierdes datos sin saberlo
❌ No verificar que la clave de join no tenga duplicados → filas duplicadas inesperadas
❌ Olvidar que los NaN después de un join NO son errores → representan datos faltantes

---

## 🚀 Siguiente paso

¡Completaste el **Nivel 1** (Fundamentos Python para ML) al 100%! 🎉

El próximo proyecto sube al **Nivel 2: Análisis Exploratorio de Datos (EDA)** — donde aprenderás a:
- Cargar un dataset real (usaremos el `dataset_completo.csv` de este proyecto)
- Calcular estadística descriptiva: media, mediana, desviación, cuartiles
- Analizar distribuciones con histogramas y boxplots
- Detectar outliers con métodos estadísticos
- Visualizar correlaciones con heatmaps y pairplots
- Generar un reporte de EDA completo

El EDA es el paso MÁS IMPORTANTE antes de construir cualquier modelo de ML. ¡Prepárate!
