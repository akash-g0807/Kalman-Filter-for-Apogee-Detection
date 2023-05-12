Array=csvread('original.csv');
col1 = Array(:, 1);
col2 = Array(:, 2);

plot(col1, col2)
hold on

Array=csvread('output.csv');
col1 = Array(:, 1);
col2 = Array(:, 2);

plot(col1, col2)
