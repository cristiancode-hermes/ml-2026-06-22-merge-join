# Learnings — 2026-06-22

## Qué se hizo
Proyecto de Merge y Join de datasets con pandas. Creamos 4 datasets realistas
de una tienda online (clientes, pedidos, productos, reseñas) con datos 
intencionalmente incompletos (clientes sin pedidos, pedidos sin reseñas) para
practicar los diferentes tipos de joins. Incluye visualizaciones comparativas
y un dashboard de análisis de negocio.

## Conceptos nuevos
- [x] INNER JOIN: solo registros que coinciden en ambas tablas
- [x] LEFT JOIN: todos de izquierda + datos de derecha (NaN si no coincide)
- [x] RIGHT JOIN: todos de derecha + datos de izquierda
- [x] OUTER JOIN: unión completa de ambas tablas
- [x] indicator=True para rastrear origen de datos en merge
- [x] pd.concat() para apilar datos vertical/horizontalmente
- [x] Encadenamiento de múltiples merges (3+ tablas)
- [x] groupby + agg después de joins para análisis

## Dificultades encontradas
- Ninguna durante la ejecución. Los datos sintéticos se generaron correctamente
  y todos los joins funcionaron como esperado.

## Notas para el próximo proyecto
- Con este proyecto completamos TODOS los skills del Nivel 1 ✅
- El próximo proyecto debe ser de Nivel 2: Análisis Exploratorio (EDA)
- Podemos usar el dataset_completo.csv generado aquí como base para EDA
- Sugerencia: EDA completo del dataset TechStore con describe(), correlaciones,
  distribuciones, heatmaps y detección de outliers
