export const MAX_FILE_SIZE = 10 * 1024 * 1024;
export function validateFile(file: File): string | null {
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || !/\.(jpe?g|png|webp)$/i.test(file.name)) return 'Choose a JPG, JPEG, PNG, or WEBP image.';
  if (file.size === 0) return 'This file is empty. Please choose another image.';
  if (file.size > MAX_FILE_SIZE) return 'This image is too large. Choose a file smaller than 10 MB.';
  return null;
}
