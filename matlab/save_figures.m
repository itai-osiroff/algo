folderPath = input("folder path?", "s");
resolution = input("specify resolution");

if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end

figHandles = findall(groot, 'Type', 'figure');

for i=1:length(figHandles)
    figHandle = figHandles(i);
    fileName = fullfile(folderPath, ['Figure_' num2str(figHandle.Number) '.png']);
    exportgraphics(figHandle, fileName, 'Resolution', resolution)
end

disp("All figures saved")
